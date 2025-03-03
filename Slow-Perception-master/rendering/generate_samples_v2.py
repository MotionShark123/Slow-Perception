import os
import sys
import json
import random
from tqdm import tqdm

# import specialized generators
from quad_angle import drawer as angle_drawer
from quad_circle1 import drawer as circle1_drawer
from quad_circle2 import drawer as circle2_drawer
from quad_circle3 import drawer as circle3_drawer
from quad_circle4 import drawer as circle4_drawer
from quad_line1 import drawer as line1_drawer
from quad_line2 import drawer as line2_drawer
from quad_point import drawer as point_drawer
from quad_point2 import drawer as point2_drawer
from quad_point3 import drawer as point3_drawer
from quad_fold import drawer as fold_drawer

# import standalone file handling functions as fallback
from gen import edit_tex_file, compile_tex_to_pdf, convert_pdf_to_png

def generate_specialized_samples(num_samples=1, output_dir="generated_samples"):
    """
    generate geometry samples using specialized quad generators
    
    args:
        num_samples: number of samples to generate
        output_dir: directory to save the generated files
    """
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # available generators with their descriptions
    generators = {
        # 'angle': angle_drawer,      # generates angle-related problems
        'circle1': circle1_drawer,  # generates circle-related problems type 1
        # 'circle2': circle2_drawer,  # generates circle-related problems type 2
        # 'circle3': circle3_drawer,  # generates circle-related problems type 3
        # 'circle4': circle4_drawer,  # generates circle-related problems type 4
        # 'line1': line1_drawer,     # generates line-related problems type 1
        # 'line2': line2_drawer,     # generates line-related problems type 2
        # 'point': point_drawer,     # generates point-related problems
        # 'point2': point2_drawer,   # generates point-related problems type 2
        # 'point3': point3_drawer,   # generates point-related problems type 3
        # 'fold': fold_drawer,       # generates folding-related problems
    }
    
    samples = []
    for i in tqdm(range(num_samples)):
        # randomly select a generator type
        # generator_type = random.choice(list(generators.keys()))
        generator_type = list(generators.keys())[i % len(generators)]
        curr_drawer = generators[generator_type]()
        
        try:
            # generate content using the selected drawer
            curr_drawer.generate_content()
            
            # save tex file
            tex_path = os.path.join(output_dir, f"sample_{i}.tex")
            
            # use class method if available, otherwise use standalone function
            if hasattr(curr_drawer, 'edit_tex_file'):
                curr_drawer.edit_tex_file(tex_path, curr_drawer.json_content)
            else:
                edit_tex_file(tex_path, curr_drawer.json_content)
            # import pdb;pdb.set_trace()
            # compile to pdf
            if hasattr(curr_drawer, 'compile_tex_to_pdf'):
                curr_drawer.compile_tex_to_pdf(tex_path)
            else:
                compile_tex_to_pdf(tex_path)
            
            os.system(f"mv sample_{i}.* {output_dir}/")
            
            # convert to png
            pdf_path = tex_path.replace('.tex', '.pdf')
            if hasattr(curr_drawer, 'convert_pdf_to_png'):
                curr_drawer.convert_pdf_to_png(pdf_path, output_dir, i)
            else:
                convert_pdf_to_png(pdf_path, output_dir, i)
            
            # clean up temporary files
            files = [f"sample_{i}.pdf", f"sample_{i}.aux", f"sample_{i}.tex", f"sample_{i}.log"]
            for f in files:
                fpath = os.path.join(output_dir, f)
                os.system(f"rm {fpath}")
            
            # save metadata
            sample_data = {
                'id': i,
                'type': generator_type,
                'content': curr_drawer.json_content,
                'image_path': f'page_{i}.png'
            }
            
            # add caption if available
            if hasattr(curr_drawer, 'json_caption'):
                sample_data['caption'] = curr_drawer.json_caption
                
            samples.append(sample_data)
            
        except Exception as e:
            print(f"error generating sample {i} with {generator_type}: {str(e)}")
            continue
    
    # save metadata to json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(samples, f, indent=4)

if __name__ == '__main__':
    generate_specialized_samples()