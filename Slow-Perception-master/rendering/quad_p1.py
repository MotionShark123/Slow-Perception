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
    angle_bisector,
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
    determine_position_by_line,
    get_locations,
    get_location,
    line_intersection
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

category = 'quad_p1'
num = 1000

class drawer:
    def __init__(self):      
        self.shape = ''
        self.types = [
            self.gen_1, # 1
            self.gen_2, # 4
            self.gen_3, # 6
            self.gen_4, # 7
            self.gen_5
        ]
        self.json_content = ''''''
        self.json_caption = '''As shown in the figure, {label5}{label6} is a line segment passing through the vertex {label1} of the square {label1}{label2}{label3}{label4}, connecting {label5}{label2}, {label5}{label4}, {label6}{label3}, and {label6}{label4}.'''
    
    def gen_1(self):
        json_content = '''\\node [{location0}, {color0}, font=\scriptsize] at ({x0},{y0}) {{{label0}}};
\\node [{location1}, {color1}, font=\scriptsize] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color2}, font=\scriptsize] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color3}, font=\scriptsize] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color4}, font=\scriptsize] at ({x4},{y4}) {{{label4}}};
\\node [{location5}, {color5}, font=\scriptsize] at ({x5},{y5}) {{{label5}}};
\\draw [line width={width}pt, {color}] ({x0},{y0}) -- ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- cycle;
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x2},{y2});
\\draw [line width={width}pt{dashed}, {color}] ({x1},{y1}) -- ({x3},{y3});
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x5},{y5});
{angle}\\coordinate ({label0}) at ({x0}, {y0});
{angle}\\coordinate ({label1}) at ({x1}, {y1});
{angle}\\coordinate ({label5}) at ({x5}, {y5});
{angle}\\tkzMarkRightAngle[size=0.2, line width={width}pt, {color}]({label0},{label5},{label1});
'''
        json_caption = '''As shown in the figure, connect the diagonals {label0}{label2} and {label1}{label3} of rectangle {label0}{label1}{label2}{label3}, intersecting at {label4}. Draw a perpendicular line through point {label0} to {label1}{label3}, with the foot of the perpendicular being {label5}.'''
        if random.uniform(0, 1) < 0.7:
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
        else:
            labels = random.sample(string.ascii_uppercase, 6)
        colors = [get_color() for _ in range(6)]        
        [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = rotate_points(generate_rectangle(), random.uniform(0, 360))
        x4, y4 = (x0 + x2) / 2, (y0 + y2) / 2
        x5, y5 = projection_point((x0, y0), (x1, y1), (x3, y3))
        locations = get_locations((x0, y0), (x1, y1), (x2, y2), (x3, y3))
        locations.append(determine_position_by_angle(LineString([(x0, y0), (x2, y2)]), LineString([(x1, y1), (x3, y3)])))
        locations.append(determine_position_by_line(LineString([(0, 0), (x5 - x0, y5 - y0)])))
        return json_content.format(
            width=random.uniform(0.5, 1),
            dashed=', dashed' if random.uniform(0, 1) > 0.2 else '',
            angle='' if random.uniform(0, 1) > 0.5 else '%',
            location0=locations[0],
            location1=locations[1],
            location2=locations[2],
            location3=locations[3],
            location4=locations[4],
            location5=locations[5],
            color=get_color(),
            color0=colors[0],
            color1=colors[1],
            color2=colors[2],
            color3=colors[3],
            color4=colors[4],
            color5=colors[5],
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            label5=labels[5],
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            x2=x2, y2=y2,
            x3=x3, y3=y3,
            x4=x4, y4=y4,
            x5=x5, y5=y5,
        ), json_caption.format(
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            label5=labels[5]
        )
    
    def gen_2(self):
        json_content = '''\\node [{location0}, {color0}, font=\scriptsize] at ({x0},{y0}) {{{label0}}};
\\node [{location1}, {color1}, font=\scriptsize] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color2}, font=\scriptsize] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color3}, font=\scriptsize] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color4}, font=\scriptsize] at ({x4},{y4}) {{{label4}}};
\\draw [line width={width}pt, {color}] ({x0},{y0}) -- ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- cycle;
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x2},{y2});
\\draw [line width={width}pt{dashed}, {color}] ({x1},{y1}) -- ({x4},{y4});
\\draw [line width={width}pt{dashed}, {color}] ({x2},{y2}) -- ({x4},{y4});
\\draw [line width={width}pt{dashed}, {color}] ({x3},{y3}) -- ({x4},{y4});
'''
        json_caption = '''As shown in the figure, connect the diagonal {label0}{label2} of trapezoid {label0}{label1}{label2}{label3}, take point {label4} so that {label0}{label2}{label4}{label3} is a parallelogram, and connect {label1}{label4}.'''
        if random.uniform(0, 1) < 0.7:
            labels = ['A', 'B', 'C', 'D', 'E']
        else:
            labels = random.sample(string.ascii_uppercase, 5)
        colors = [get_color() for _ in range(5)]        
        [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = rotate_points(generate_trapezoid(), random.uniform(0, 360))
        x4, y4 = x2 + x3 - x0, y2 + y3 - y0
        locations = get_locations((x0, y0), (x1, y1), (x2, y2), (x3, y3))
        locations.append(get_location((-x2, -y2), (-x4, -y4), (-x3, -y3)))
        return json_content.format(
            width=random.uniform(0.5, 1),
            dashed=', dashed' if random.uniform(0, 1) > 0.8 else '',
            location0=locations[0],
            location1=locations[1],
            location2=locations[2],
            location3=locations[3],
            location4=locations[4],
            color=get_color(),
            color0=colors[0],
            color1=colors[1],
            color2=colors[2],
            color3=colors[3],
            color4=colors[4],
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            x2=x2, y2=y2,
            x3=x3, y3=y3,
            x4=x4, y4=y4,
        ), json_caption.format(
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
        )
    
    def gen_3(self):
        json_content = '''\\node [{location0}, {color0}, font=\scriptsize] at ({x0},{y0}) {{{label0}}};
\\node [{location1}, {color1}, font=\scriptsize] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color2}, font=\scriptsize] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color3}, font=\scriptsize] at ({x3},{y3}) {{{label3}}};
%\\node [{location4}, {color4}, font=\scriptsize] at ({x4},{y4}) {{{label4}}};
\\node [{location5}, {color5}, font=\scriptsize] at ({x5},{y5}) {{{label5}}};
\\node [{location6}, {color6}, font=\scriptsize] at ({x6},{y6}) {{{label6}}};
\\node [{location7}, {color7}, font=\scriptsize] at ({x7},{y7}) {{{label7}}};
\\node [{location8}, {color8}, font=\scriptsize] at ({x8},{y8}) {{{label8}}};
\\draw [line width={width}pt, {color}] ({x0},{y0}) -- ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- cycle;
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x2},{y2});
\\draw [line width={width}pt{dashed}, {color}] ({x1},{y1}) -- ({x3},{y3});
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x5},{y5});
\\draw [line width={width}pt{dashed}, {color}] ({x1},{y1}) -- ({x6},{y6});
\\draw [line width={width}pt{dashed}, {color}] ({x2},{y2}) -- ({x7},{y7});
\\draw [line width={width}pt{dashed}, {color}] ({x3},{y3}) -- ({x8},{y8});
\\draw [line width={width}pt, {color}] ({x5},{y5}) -- ({x6},{y6}) -- ({x7},{y7}) -- ({x8},{y8}) -- cycle;
'''
        json_caption = '''As shown in the figure, from each vertex of the parallelogram {label0}{label1}{label2}{label3}, draw perpendicular lines {label0}{label5}, {label1}{label6}, {label2}{label7}, and {label3}{label8} to the diagonals. The feet of the perpendicular lines are {label5}, {label6}, {label7}, and {label8} respectively.'''
        if random.uniform(0, 1) < 0.7:
            labels = ['A', 'B', 'C', 'D', 'O', 'E', 'F', 'G', 'H']
        else:
            labels = random.sample(string.ascii_uppercase, 9)
        colors = [get_color() for _ in range(9)]        
        [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = rotate_points(generate_parallelogram(), random.uniform(0, 360))
        x4, y4 = (x0 + x2) / 2, (y0 + y2) / 2
        (x5, y5) = projection_point((x0, y0), (x1, y1), (x3, y3))
        (x6, y6) = projection_point((x1, y1), (x0, y0), (x2, y2))
        (x7, y7) = projection_point((x2, y2), (x1, y1), (x3, y3))
        (x8, y8) = projection_point((x3, y3), (x0, y0), (x2, y2))
        locations = get_locations((x0, y0), (x1, y1), (x2, y2), (x3, y3))
        locations.append(get_location((x0, y0), (x4, y4), (x1, y1)))
        locations.append(determine_position_by_line(LineString([(0, 0), (x5 - x0, y5 - y0)])))
        locations.append(determine_position_by_line(LineString([(0, 0), (x6 - x1, y6 - y1)])))
        locations.append(determine_position_by_line(LineString([(0, 0), (x7 - x2, y7 - y2)])))
        locations.append(determine_position_by_line(LineString([(0, 0), (x8 - x3, y8 - y3)])))
        return json_content.format(
            width=random.uniform(0.5, 1),
            dashed=', dashed' if random.uniform(0, 1) > 0.8 else '',
            location0=locations[0],
            location1=locations[1],
            location2=locations[2],
            location3=locations[3],
            location4=locations[4],
            location5=locations[5],
            location6=locations[6],
            location7=locations[7],
            location8=locations[8],
            color=get_color(),
            color0=colors[0],
            color1=colors[1],
            color2=colors[2],
            color3=colors[3],
            color4=colors[4],
            color5=colors[5],
            color6=colors[6],
            color7=colors[7],
            color8=colors[8],
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            label5=labels[5],
            label6=labels[6],
            label7=labels[7],
            label8=labels[8],
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            x2=x2, y2=y2,
            x3=x3, y3=y3,
            x4=x4, y4=y4,
            x5=x5, y5=y5,
            x6=x6, y6=y6,
            x7=x7, y7=y7,
            x8=x8, y8=y8
        ), json_caption.format(
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            label5=labels[5],
            label6=labels[6],
            label7=labels[7],
            label8=labels[8],
        )
    
    def gen_4(self):
        json_content = '''\\node [{location0}, {color0}, font=\scriptsize] at ({x0},{y0}) {{{label0}}};
\\node [{location1}, {color1}, font=\scriptsize] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color2}, font=\scriptsize] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color3}, font=\scriptsize] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color4}, font=\scriptsize] at ({x4},{y4}) {{{label4}}};
\\node [{location5}, {color5}, font=\scriptsize] at ({x5},{y5}) {{{label5}}};
\\draw [line width={width}pt, {color}] ({x0},{y0}) -- ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- cycle;
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x2},{y2});
\\draw [line width={width}pt{dashed}, {color}] ({x1},{y1}) -- ({x3},{y3});
\\draw [line width={width}pt{dashed}, {color}] ({x1},{y1}) -- ({x5},{y5});
\\draw [line width={width}pt{dashed}, {color}] ({x2},{y2}) -- ({x5},{y5});
\\draw [line width={width}pt{dashed}, {color}] ({x4},{y4}) -- ({x5},{y5});
'''
        json_caption = '''As shown in the figure, the intersection point of the diagonal lines of trapezoid {label0}{label1}{label2}{label3} is {label4}. Take a point {label5} on the extension line of {label0}{label1} and connect {label5}{label1} and {label5}{label2}.'''
        if random.uniform(0, 1) < 0.7:
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
        else:
            labels = random.sample(string.ascii_uppercase, 6)
        colors = [get_color() for _ in range(6)]        
        [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = rotate_points(generate_trapezoid(), random.uniform(0, 360))
        x4, y4 = line_intersection((x0, y0), (x2, y2), (x1, y1), (x3, y3))
        i = random.uniform(0.3, 0.7)
        x5, y5 = x1 + (x1 - x0) * i, y1 + (y1 - y0) * i
        locations = get_locations((x0, y0), (x1, y1), (x2, y2), (x3, y3))
        locations.append(determine_position_by_angle(LineString([(x0, y0), (x2, y2)]), LineString([(x1, y1), (x3, y3)])))
        locations.append(get_location((-x2, -y2), (-x5, -y5), (-x3, -y3)))
        return json_content.format(
            width=random.uniform(0.5, 1),
            dashed=', dashed' if random.uniform(0, 1) > 0.8 else '',
            location0=locations[0],
            location1=locations[1],
            location2=locations[2],
            location3=locations[3],
            location4=locations[4],
            location5=locations[5],
            color=get_color(),
            color0=colors[0],
            color1=colors[1],
            color2=colors[2],
            color3=colors[3],
            color4=colors[4],
            color5=colors[5],
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            label5=labels[5],
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            x2=x2, y2=y2,
            x3=x3, y3=y3,
            x4=x4, y4=y4,
            x5=x5, y5=y5
        ), json_caption.format(
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            label5=labels[5]
        )
    
    def gen_5(self):
        json_content = '''\\node [{location0}, {color0}, font=\scriptsize] at ({x0},{y0}) {{{label0}}};
\\node [{location1}, {color1}, font=\scriptsize] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color2}, font=\scriptsize] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color3}, font=\scriptsize] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color4}, font=\scriptsize] at ({x4},{y4}) {{{label4}}};
\\node [{location5}, {color5}, font=\scriptsize] at ({x5},{y5}) {{{label5}}};
\\node [{location6}, {color6}, font=\scriptsize] at ({x6},{y6}) {{{label6}}};
\\node [{location7}, {color7}, font=\scriptsize] at ({x7},{y7}) {{{label7}}};
\\draw [line width={width}pt, {color}] ({x0},{y0}) -- ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- cycle;
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x2},{y2});
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x6},{y6});
\\draw [line width={width}pt{dashed}, {color}] ({x0},{y0}) -- ({x7},{y7});
\\draw [line width={width}pt{dashed}, {color}] ({x3},{y3}) -- ({x5},{y5});
\\draw [line width={width}pt{dashed}, {color}] ({x4},{y4}) -- ({x5},{y5});
\\draw [line width={width}pt{dashed}, {color}] ({x6},{y6}) -- ({x7},{y7});
'''
        json_caption = '''As shown in the figure, in the square {label0}{label1}{label2}{label3}, {label4}{label5} is parallel to {label0}{label2}, connect {label4}{label5}, {label3}{label5}, take point {label7} on {label5}{label3}, connect {label7}{label4} and extend, intersecting the extended line of {label3}{label0} at {label6}'''
        if random.uniform(0, 1) < 0.7:
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        else:
            labels = random.sample(string.ascii_uppercase, 8)
        colors = [get_color() for _ in range(8)]        
        [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = rotate_points(generate_square(), random.uniform(0, 360))
        i = random.uniform(0.3, 0.4)
        x4, y4 = x0 * i + x1 * (1 - i), y0 * i + y1 * (1 - i)
        x5, y5 = x2 * i + x1 * (1 - i), y2 * i + y1 * (1 - i)
        i = random.uniform(0.05, 0.15)
        x7, y7 = x3 * i + x5 * (1 - i), y3 * i + y5 * (1 - i)
        x6, y6 = line_intersection((x0, y0), (x3, y3), (x4, y4), (x7, y7))
        locations = get_locations((x0, y0), (x1, y1), (x2, y2), (x3, y3))
        locations.append(get_location((x1, y1), (x4, y4), (x6, y6)))
        locations.append(get_location((-x4, -y4), (-x5, -y5), (-x3, -y3)))
        locations.append(get_location((-x0, -y0), (-x6, -y6), (-x4, -y4)))
        locations.append(get_location((-x4, -y4), (-x7, -y7), (-x0, -y0)))
        return json_content.format(
            width=random.uniform(0.5, 1),
            dashed=', dashed' if random.uniform(0, 1) > 0.8 else '',
            location0=locations[0],
            location1=locations[1],
            location2=locations[2],
            location3=locations[3],
            location4=locations[4],
            location5=locations[5],
            location6=locations[6],
            location7=locations[7],
            color=get_color(),
            color0=colors[0],
            color1=colors[1],
            color2=colors[2],
            color3=colors[3],
            color4=colors[4],
            color5=colors[5],
            color6=colors[6],
            color7=colors[7],
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            label5=labels[5],
            label6=labels[6],
            label7=labels[7],
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            x2=x2, y2=y2,
            x3=x3, y3=y3,
            x4=x4, y4=y4,
            x5=x5, y5=y5,
            x6=x6, y6=y6,
            x7=x7, y7=y7,
        ), json_caption.format(
            label0=labels[0],
            label1=labels[1],
            label2=labels[2],
            label3=labels[3],
            label4=labels[4],
            label5=labels[5],
            label6=labels[6],
            label7=labels[7],
        )
    
    def generate_content(self):
        self.json_content, self.json_caption = self.types[random.randint(0, 4)]()

def process_task(index):
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
    os.system(f'rm -rf texput.log')
