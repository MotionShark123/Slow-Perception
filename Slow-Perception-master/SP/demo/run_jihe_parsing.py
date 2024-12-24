import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from SP.utils.conversation import conv_templates, SeparatorStyle
from SP.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from SP.model import *
from SP.utils.utils import KeywordsStoppingCriteria
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
import glob
from transformers import TextStreamer
from SP.model.plug.transforms import train_transform, test_transform
import json


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'




def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def pad_to_square(image):
    width, height = image.size
    if width > height:
        padding = (0, (width - height) // 2, 0, (width - height) - (width - height) // 2)
    else:
        padding = ((height - width) // 2, 0, (height - width) - (height - width) // 2, 0)
    square_image = ImageOps.expand(image, padding, fill='white')
    return square_image

def add_outer_padding(image):
    width, height = image.size
    border_size = width // 5
    padded_image = ImageOps.expand(image, border=border_size, fill='white')
    return padded_image


def process_image(image):

    square_image = pad_to_square(image)
    padded_image = add_outer_padding(square_image)

    return padded_image

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = SPQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()

    model.to(device='cuda',  dtype=torch.float16)

    image_processor = test_transform
    image_processor_high =  test_transform
    use_im_start_end = True



    image_token_len = 256


    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    qs = ''


    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs



    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()



    inputs = tokenizer([prompt])


    if '.jpg' in args.image_file or '.png' in args.image_file:

        image = load_image(args.image_file)

        image = process_image(image)
        image_1 = image.copy()

        image_tensor = image_processor(image)
        image_tensor_1 = image_processor_high(image_1)

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


        with torch.autocast("cuda", dtype=torch.float16):
            output_ids = model.generate(
                input_ids,
                images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                do_sample=False,
                num_beams = 1,
                # temperature=0.5,
                streamer=streamer,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria]
                )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()


        outputs_think = outputs

        if 'Circle' not in outputs_think:

            outputs_think_p = outputs_think.split('Line:\n')[1]
            outputs_think_c = ''
        else:
            outputs_think_p = outputs_think.split('Line:\n')[1][:-1]
            outputs_think_c = outputs_think.split('Line:\n')[1].split('Circle:\n')[1]
            if outputs_think_c[-1] == '\n':
                outputs_think_c = outputs_think_c[:-1]

        # # print(outputs_think_p)
        outputs_think_p_list = outputs_think_p.split('\n')
        outputs_think_c_list = outputs_think_c.split('\n')
        # print(outputs_think_p_list)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        line_list = []
        # line_real_list = []
        for p in outputs_think_p_list:
            # try:
            if '--' in p:
                # if 'cycle' not in p:
                try:
                    p = p.split(': ')[1]
                except:
                    pass

                p0 = eval(p.split(' -- ')[0])
                p1 = eval(p.split(' -- ')[-1])

                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 3, color = 'k')

                ax.scatter(p0[0], p0[1], s=10)
                ax.scatter(p1[0], p1[1], s=10)


                # p_list= p.split(' -- ')
                # colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
                # # colors = ['green', 'blue', 'indigo', 'violet']
                # line_p_new = []
                # for pp_idx in range(len(p_list) -1):
                #     color = colors[pp_idx]
                #     p0_s = eval(p_list[pp_idx])
                #     p1_s = eval(p_list[pp_idx+1])
                #     # ax.plot([p0_s[0], p1_s[0]], [p0_s[1], p1_s[1]], color = color, linewidth = 3)




        # print(line_list)
        for c in outputs_think_c_list:
            if c:
                # c = c.split(': ')[1]
                c = eval(c)
                r = float(c[2])
                center = (c[0], c[1])
                circle = plt.Circle(center, r, linewidth = 3, color = 'k',  fill=False)
                ax.add_artist(circle)
                # ax2.add_artist(circle)

        plt.savefig('results/demo.png', bbox_inches='tight', pad_inches=0.1)
        # plt.savefig('/data/show/figure.png', bbox_inches='tight', pad_inches=0.1)


    else:
        files = glob.glob(args.image_file + '/*')

        out_list = []
        for file in tqdm(files):

            image = load_image(file)
            image_1 = image.copy()
            # image_1 = image_1.resize((1024, 1024))
            image_tensor = image_processor(image)

            out_path = 'results/' + file.split('/')[-1]
            image_tensor_1 = image_processor_high(image_1)

            out_dict = {}

            name =  file.split('/')[-1]

            # out_dict[]
            # print(image_tensor_1.shape)

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)




            # print(f"{roles[1]}: ", end="")
            with torch.autocast("cuda", dtype=torch.float16):
                output_ids = model.generate(
                    input_ids,
                    images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                    do_sample=False,
                    num_beams = 1,
                    # temperature=0.2,
                    streamer=streamer,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria]
                    )

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs_think = outputs.strip()

            # print(outputs_think)
            if 'Circle' not in outputs_think:
                outputs_think_p = outputs_think.split('Line:\n')[1]
                if outputs_think_p[-1] == '\n':
                    outputs_think_p = outputs_think_p[:-1]
                # outputs_think_c_list = []
                outputs_think_c = ''
            else:
                outputs_think_p = outputs_think.split('Circle:\n')[0].split('Line:\n')[1]
                if outputs_think_p[-1] == '\n':
                    outputs_think_p = outputs_think_p[:-1]
                outputs_think_c = outputs_think.split('Line:\n')[1].split('Circle:\n')[1]
                if outputs_think_c[-1] == '\n':
                    outputs_think_c = outputs_think_c[:-1]

            outputs_think_p_list = outputs_think_p.split('\n')
            outputs_think_c_list = outputs_think_c.split('\n')
            # print(outputs_think_p_list)
            fig, ax = plt.subplots(figsize=(10,10))
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

            line_list = []
            for p in outputs_think_p_list:
                # try:
                if '--' in p:


                    p0 = eval(p.split(' -- ')[0])
                    p1 = eval(p.split(' -- ')[-1])
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]])
                    # print([p0, p1])
                    ax.scatter(p0[0], p0[1], s=10)
                    ax.scatter(p1[0], p1[1], s=10)

                    line_list.append(((p0[0], p0[1]), (p1[0], p1[1])))


            out_dict[name] = line_list

            out_list.append(out_dict)
            for c in outputs_think_c_list:
                if c:
                    c = eval(c)
                    r = float(c[2])
                    center = (c[0], c[1])
                    circle = plt.Circle(center, r, fill=False)
                    ax.add_artist(circle)

            plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)

        filename = 'results/slow_perception.json'
        with open(filename, 'w', encoding="utf-8") as file_obj:
            json.dump(out_list, file_obj, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
