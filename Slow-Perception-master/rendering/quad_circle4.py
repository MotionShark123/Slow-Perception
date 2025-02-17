import random
import string
import os
from tqdm import tqdm
import json
from multiprocessing import Process
import re
import argparse 
import numpy as np
from shapely.geometry import LineString
from geo import (
    get_shape,
    get_color,
    distance, 
    area_of_quadrilateral, 
    incenter, 
    generate_quadrilateral_with_incircle, 
    sort_points_clockwise, 
    generate_parallelogram, 
    generate_rectangle, 
    generate_rhombus, 
    generate_trapezoid, 
    generate_square, 
    generate_right_trapezoid, 
    generate_isosceles_trapezoid, 
    projection_point, 
    determine_fifth_point_position, 
    is_close, 
    find_intersections_excluding_endpoints, 
    get_angle, 
    determine_position_by_angle, 
    rotate_points,
    determine_position_by_line
)
from gen import (
    edit_tex_file,
    compile_tex_to_pdf,
    convert_pdf_to_png
)
from edit import (
    process_value,
    process_conversations,
    process_json
)

category = 'quad_circle4'

class drawer:
    def __init__(self):      
        self.shape = ''
        self.json_content = '''\\filldraw ({x0},{y0}) circle (0.5pt);
\\node [{location}, {color}, font=\scriptsize] at ({x0}, {y0}) {{{label}}};
\\node [{location1}, {color2}, font=\scriptsize] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color3}, font=\scriptsize] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color4}, font=\scriptsize] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color5}, font=\scriptsize] at ({x4},{y4}) {{{label4}}};
\\node [{location5}, {color1}, font=\scriptsize] at ({x5},{y5}) {{{label5}}};
\\draw [line width={width1}pt] ({x0}, {y0}) circle ({r});
\\draw [line width={width2}pt, {color1}] ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- ({x4},{y4}) -- cycle;
\\draw [line width={width2}pt, {color1}] ({x1},{y1}) -- ({x3},{y3});
\\draw [line width={width2}pt, {color1}] ({x2},{y2}) -- ({x4},{y4});
\\draw [line width={width2}pt{dashed}, {color1}] ({x3},{y3}) -- ({x5},{y5});
{fill}\\fill[{colorfill1}, opacity=0.2] ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- cycle;
{fill}\\fill[{colorfill2}, opacity=0.2] ({x3},{y3}) -- ({x4},{y4}) -- ({x5},{y5}) -- cycle;
'''
        self.json_caption = '''As shown in the figure, circle {label} is the circumcircle of quadrilateral {label1}{label2}{label3}{label4}. Connect {label1}{label3} and {label2}{label4}, and take point {label5} on {label2}{label4} so that triangle {label4}{label5}{label3} is similar to triangle {label1}{label2}{label3}.'''
   
    def generate_content(self):
        width = random.uniform(0.5, 1)
        colors = [get_color() for _ in range(6)]        
        coordinates = generate_quadrilateral_with_incircle()
        coordinates = rotate_points(coordinates, random.uniform(0, 360))
        if random.uniform(0, 1) < 0.7:
            labels = ['A', 'B', 'C', 'D', 'O', 'E']
        else:
            labels = random.sample(string.ascii_uppercase, 6)
        (x0, y0), r = incenter(coordinates[0], coordinates[1], coordinates[2], coordinates[3]) 
        (x1, y1) = projection_point((x0, y0), coordinates[0], coordinates[1])
        (x2, y2) = projection_point((x0, y0), coordinates[1], coordinates[2])
        (x3, y3) = projection_point((x0, y0), coordinates[2], coordinates[3])
        (x4, y4) = projection_point((x0, y0), coordinates[3], coordinates[0])
        CD = distance((x3, y3), (x4, y4))
        AB = distance((x1, y1), (x2, y2))
        AC = distance((x1, y1), (x3, y3))
        i = CD / AC * AB
        x5, y5 = x4 + (x2 - x4) / np.linalg.norm((x4 - x2, y4 - y2)) * i, y4 + (y2 - y4) / np.linalg.norm((x4 - x2, y4 - y2)) * i
        location=random.choice(['above right', 'above', 'above left', 'left', 'below left', 'below', 'below right', 'right'])
        locations = []
        for _ in range(4):
            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            points.append(points[_])
            locations.append(determine_fifth_point_position(points))
        (x, y) = projection_point((x3, y3), (x2, y2), (x4, y4))
        locations.append(determine_position_by_line(LineString([(x3, y3), (x, y)])))
        colorfill1 = random.choice([
            "red", "green", "blue", "cyan", "magenta", "yellow", "orange",
            "purple", "brown", "pink", "lime", "olive", "teal", "violet", "gray"
        ]) if random.uniform(0, 1) < 0.6 else "black"
        colorfill2 = random.choice([
            "red", "green", "blue", "cyan", "magenta", "yellow", "orange",
            "purple", "brown", "pink", "lime", "olive", "teal", "violet", "gray"
        ]) if colorfill1 != 'black' else "black"
        self.json_content = self.json_content.format(
            width1=width + 0.2,
            width2=width,
            location=location,
            location1=locations[0],
            location2=locations[1],
            location3=locations[2],
            location4=locations[3],
            location5=locations[4],
            color1=colors[0],
            color2=colors[1],
            color3=colors[2],
            color4=colors[3],
            color5=colors[4],
            color=colors[5],
            label1=labels[0],
            label2=labels[1],
            label3=labels[2],
            label4=labels[3],
            label=labels[4],
            label5=labels[5],
            x0=x0, y0=y0, r=r,
            x1=x1, y1=y1,
            x2=x2, y2=y2,
            x3=x3, y3=y3,
            x4=x4, y4=y4,
            x5=x5, y5=y5,
            dashed=', dashed' if random.uniform(0, 1) > 0.2 else '',
            fill='%' if random.uniform(0, 1) > 0.5 else '',
            colorfill1=colorfill1,
            colorfill2=colorfill2
        )
        self.json_caption = self.json_caption.format(
            label1=labels[0],
            label2=labels[1],
            label3=labels[2],
            label4=labels[3],
            label=labels[4],
            label5=labels[5]
        )

def process_task(index):
    num = 200
    success = 0
    image_fold = f'./{category}/quad_images_{index}'
    image_fold_name = f'quad_images_{index}'
    if not os.path.exists(image_fold):
        os.makedirs(image_fold)
    new_json_file = f'./{category}_{index}.json'
    all_list = []
    pdar = tqdm(total=num, desc=f'Curr_process{index} is processing')
    while (success < num):
        new_tex_file = f'./curr_{index}.tex'
        new_pdf_file = f'./curr_{index}.pdf'
        curr_image = drawer()
        curr_image.generate_content()
        edit_tex_file(new_tex_file, curr_image.json_content)
        compile_tex_to_pdf(new_tex_file)
        convert_pdf_to_png(new_pdf_file, f'{image_fold}', success)
        newdict = {
            'id': success,
            'image': f'{category}/{image_fold_name}/page_{success}.png',
            'conversations': [{
                    'from': 'human',
                    'value': '<image>\nOCR with format: '
                }, {
                    'from': 'gpt',
                    'value': '\\begin{tikzpicture}\n' + curr_image.json_content + '\n\\end{tikzpicture}'
                }
            ],
            'caption': curr_image.json_caption
        }
        all_list.append(newdict)
        pdar.update(1)
        success += 1
    with open(new_json_file, 'w') as json_file:
        json.dump(all_list, json_file, indent=4)

def main(args):
    curr_index = args.curr_index
    process_num = 10
    processes = []
    for i in range(process_num): 
        p = Process(target=process_task, args=((i % process_num + curr_index * process_num),))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def extract_number(file_path):
    filename = os.path.basename(file_path)
    match = pattern.match(filename)
    if match:
        return int(match.group(1))
    return float('inf')

def gen():
    curr_image = drawer()
    curr_image.generate_content()
    return curr_image.json_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing.')
    parser.add_argument('--curr_index', type=int, help='curr gpu id', default=0)
    args = parser.parse_args()
    main(args)
    current_directory = os.getcwd()
    pattern = re.compile(fr'^{category}_(\d+)\.json$')
    file_paths = [
        os.path.join(current_directory, filename)
        for filename in os.listdir(current_directory)
        if pattern.match(filename)
    ]
    file_paths.sort(key=extract_number)
    merged_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)
    with open(f'{category}_pre.json', 'w') as output_file:
        json.dump(merged_data, output_file, indent=4)
    os.system('rm -rf curr*')
    input_file = f'{category}_pre.json'
    output_file = f'{category}.json'
    process_json(input_file, output_file)
    os.system(f'rm -rf {category}_*.json')
