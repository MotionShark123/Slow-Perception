import os
import sys
import json
import random
from tqdm import tqdm

from geo import (
    get_shape,
    get_color,
    generate_parallelogram,
    generate_rectangle,
    generate_rhombus,
    generate_trapezoid,
    generate_square,
    generate_right_trapezoid,
    generate_isosceles_trapezoid,
    generate_quadrilateral_with_incircle
)
from gen import (
    edit_tex_file,
    compile_tex_to_pdf,
    convert_pdf_to_png
)

def generate_shape_points(shape_type):
    """
    generate points for a specific shape type using existing functions
    
    args:
        shape_type: string indicating which shape to generate
    """
    shape_generators = {
        'Parallelogram': generate_parallelogram,
        'Rectangle': generate_rectangle,
        'Rhombus': generate_rhombus,
        'Trapezoid': generate_trapezoid,
        'Square': generate_square,
        'Right Trapezoid': generate_right_trapezoid,
        'Isosceles Trapezoid': generate_isosceles_trapezoid,
        'Others': generate_quadrilateral_with_incircle
    }
    
    generator = shape_generators.get(shape_type)
    if generator:
        return generator()
    return None

def generate_samples(num_samples=8, output_dir="generated_samples"):
    """
    generate random geometry samples using existing functionality
    
    args:
        num_samples: number of samples to generate
        output_dir: directory to save the generated files
    """
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # available shapes (with weights as defined in geo.py's get_shape function)
    shapes = [
        'Parallelogram',    # 平行四边形
        'Rectangle',        # 矩形
        'Rhombus',         # 菱形
        'Trapezoid',       # 梯形
        'Square',          # 正方形
        'Right Trapezoid', # 直角梯形
        'Isosceles Trapezoid',  # 等腰梯形
        'Others'
    ]
    
    # weights from geo.py's get_shape function
    weights = [0, 1, 0, 0, 0, 0, 0, 0]
    
    samples = []
    for i in tqdm(range(num_samples)):
        # randomly select shape type using the same weights as in geo.py
        # shape_type = random.choices(shapes, weights=weights, k=1)[0]
        shape_type = shapes[i % len(shapes)]
        color = get_color()
        
        # generate points for the selected shape
        points = generate_shape_points(shape_type)
        if points is None:
            continue
            
        # convert points to the right format if needed
        if hasattr(points, 'tolist'):
            points = points.tolist()
        
        # create tex content
        tex_content = f'''
        \\draw [thick, {color}] ({points[0][0]},{points[0][1]}) -- ({points[1][0]},{points[1][1]}) -- 
                                ({points[2][0]},{points[2][1]}) -- ({points[3][0]},{points[3][1]}) -- cycle;
        \\node [above] at ({points[0][0]},{points[0][1]}) {{A}};
        \\node [right] at ({points[1][0]},{points[1][1]}) {{B}};
        \\node [below] at ({points[2][0]},{points[2][1]}) {{C}};
        \\node [left] at ({points[3][0]},{points[3][1]}) {{D}};
        '''
        
        # save tex file
        tex_path = os.path.join(output_dir, f"sample_{i}.tex")
        edit_tex_file(tex_path, tex_content)
        
        # compile to pdf
        compile_tex_to_pdf(tex_path)
        os.system(f"mv sample_{i}.* {output_dir}/")

        # convert to png
        pdf_path = tex_path.replace('.tex', '.pdf')
        convert_pdf_to_png(pdf_path, output_dir, i)
        files = [f"sample_{i}.pdf", f"sample_{i}.aux", f"sample_{i}.tex", f"sample_{i}.log"]
        for f in files:
            fpath = os.path.join(output_dir, f)
            os.system(f"rm {fpath}")
        
        # save metadata
        sample_data = {
            'id': i,
            'shape_type': shape_type,
            'color': color,
            'points': points,
            'image_path': f'page_{i}.png'
        }
        samples.append(sample_data)
    
    # save metadata to json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(samples, f, indent=4)

if __name__ == '__main__':
    generate_samples()