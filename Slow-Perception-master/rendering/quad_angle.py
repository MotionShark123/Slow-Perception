import sys

sys.path.append('/home/m-yinyouyang/.local/lib/python3.8/site-packages')

import os
import random
import string
import json
import argparse 
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path
import pdf2image.exceptions as pp
from multiprocessing import Process
import math
from scipy.spatial import ConvexHull


class drawer:
    def __init__(self):
        self.shape = ''
        self.shapes = [
            'Parallelogram',    # 平行四边形
            'Rectangle',        # 矩形
            'Rhombus',          # 菱形
            'Trapezoid',        # 梯形
            'Square',           # 正方形
            'Right Trapezoid',  # 直角梯形
            'Isosceles Trapezoid',  # 等腰梯形
            'Others', 
        ] 
        self.types = [
            'Angle'
            # 'Fold'
        ]
        self.funcs = {
            'Angle': self.func_1
            # 'Fold': self.func_2
        }
        self.colors = ['black', 'blue', 'red', 'green', 'yellow', 'orange', 'violet', 'pink', 'cyan']
        self.new_content = '''\\draw [thick, {color1}] ({x1},{y1}) -- ({x2},{y2});
\\draw [thick, {color1}] ({x2},{y2}) -- ({x3},{y3});
\\draw [thick, {color1}] ({x3},{y3}) -- ({x4},{y4});
\\draw [thick, {color1}] ({x4},{y4}) -- ({x1},{y1});
\\node [{location1}, {color2}] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color3}] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color4}] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color5}] at ({x4},{y4}) {{{label4}}};
'''
# {length1_vis}\\node at [{location12}] ({x12},{y12}) {{{label12}}};
# {length2_vis}\\node at [{location23}] ({x23},{y23}) {{{label23}}};
# {length3_vis}\\node at [{location34}] ({x34},{y34}) {{{label34}}};
# {length4_vis}\\node at [{location41}] ({x41},{y41}) {{{label41}}};
# '''
        self.json_content = '''\\draw [thick, {color1}] ({x1},{y1}) -- ({x2},{y2});
\\draw [thick, {color1}] ({x2},{y2}) -- ({x3},{y3});
\\draw [thick, {color1}] ({x3},{y3}) -- ({x4},{y4});
\\draw [thick, {color1}] ({x4},{y4}) -- ({x1},{y1});
\\node [{location1}, {color2}] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color3}] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color4}] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color5}] at ({x4},{y4}) {{{label4}}};
'''
# {length1_vis}\\node at [{location12}] ({x12},{y12}) {{{label12}}};
# {length2_vis}\\node at [{location23}] ({x23},{y23}) {{{label23}}};
# {length3_vis}\\node at [{location34}] ({x34},{y34}) {{{label34}}};
# {length4_vis}\\node at [{location41}] ({x41},{y41}) {{{label41}}};
# '''

    def get_shape(self):
        return random.choices(self.shapes, weights=[0.5, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0], k=1)[0];

    def get_color(self):
        return random.choices(self.colors, weights=[0.96, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005], k=1)[0]

    def generate_parallelogram(self):
        x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
        x2, y2 = x1 + random.uniform(3, 10), y1
        x3, y3 = x1 + random.uniform(3, 5) if random.uniform(0, 1) < 0.5 else x1 - random.uniform(3, 5), y1 + random.uniform(3, 10)
        x4 = x1 + x3 - x2
        y4 = y1 + y3 - y2
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    def generate_rectangle(self):
        x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
        width = random.uniform(3, 10)
        height = random.uniform(3, 10)
        x2, y2 = x1 + width, y1
        x3, y3 = x1 + width, y1 + height
        x4, y4 = x1, y1 + height
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    def generate_rhombus(self):
        x0, y0 = random.uniform(-10, 10), random.uniform(-10, 10)
        width = random.uniform(3, 5)
        height = random.uniform(7, 10)
        x1, y1 = x0 - width / 2, y0
        x2, y2 = x0, y0 - height / 2
        x3, y3 = x0 + width / 2, y0
        x4, y4 = x0, y0 + height / 2
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    def generate_trapezoid(self):
        x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
        base_width = random.uniform(3, 10)
        top_width = random.uniform(3, 10)
        height = random.uniform(3, 10)
        x2, y2 = x1 + base_width, y1
        x3, y3 = random.uniform(-10, 10), y1 + height
        x4, y4 = x3 - top_width, y1 + height
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    def generate_square(self):
        x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
        size = random.uniform(3, 10)
        x2, y2 = x1 + size, y1
        x3, y3 = x1 + size, y1 + size
        x4, y4 = x1, y1 + size
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    def generate_right_trapezoid(self):
        x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
        base_width = random.uniform(3, 10)
        top_width = random.uniform(3, 10)
        height = random.uniform(3, 10)
        x2, y2 = x1 + base_width, y1
        x3, y3 = x1 + top_width, y1 + height
        x4, y4 = x1, y1 + height
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    def generate_isosceles_trapezoid(self):
        x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
        base_width = random.uniform(3, 10)
        height = random.uniform(3, 10)
        top_width = random.uniform(1, base_width)
        x2, y2 = x1 + base_width, y1
        x3, y3 = x1 + base_width - (base_width - top_width) / 2, y1 + height
        x4, y4 = x1 + (base_width - top_width) / 2, y1 + height
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    def projection_point(self, A, B, C):
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        BC = C - B
        BA = A - B
        BC_unit = BC / np.linalg.norm(BC)
        projection_length = np.dot(BA, BC_unit)
        projection_vector = projection_length * BC_unit
        D = B + projection_vector
        return tuple(D)

    def determine_fifth_point_position(self, points):
        hull = ConvexHull(points)
        hull_points = np.array(points)[hull.vertices]
        E_index = np.where(np.all(hull_points == points[4], axis=1))[0]
        if len(E_index) == 0:
            min_distance = np.inf
            closest_edge = None
            for i in range(len(hull_points)):
                A = hull_points[i]
                B = hull_points[(i + 1) % len(hull_points)]
                AB = B - A
                EB = points[4] - B
                cross_product = np.cross(AB, EB)
                AB_length = np.linalg.norm(AB)
                distance = np.abs(cross_product) / AB_length
                if distance < min_distance:
                    closest_edge = (A, B, i)
                    min_distance = distance
            if closest_edge:
                A, B, i = closest_edge
                AB = np.array(B) - np.array(A)
                vec = np.array([-AB[1], AB[0]])
                angle_bisector = vec
                C = hull_points[(i + 2) % len(hull_points)] - points[4]
                if np.dot(C, vec) > 0:
                    angle_bisector *= -1 
            else:
                return 'auto'
        else:
            E_index = E_index[0]
            X = hull_points[(E_index - 1) % len(hull_points)]
            Y = hull_points[(E_index + 1) % len(hull_points)]
            vector_XE = X - points[4]
            vector_EY = Y - points[4]
            angle_bisector = vector_XE / np.linalg.norm(vector_XE) + vector_EY / np.linalg.norm(vector_EY)
            angle_bisector *= -1
        x5, y5 = angle_bisector
        if x5 >= 0 and y5 >= 0:
            return 'above right'
        elif x5 <= 0 and y5 >= 0:
            return 'above left'
        elif x5 <= 0 and y5 <= 0:
            return 'below left'
        elif x5 >= 0 and y5 <= 0:
            return 'below right'
    
    def calculate_angle(self, A, B, C):
        x1, y1 = A
        x2, y2 = B
        x3, y3 = C
        BA = (x1 - x2, y1 - y2)
        BC = (x3 - x2, y3 - y2)
        dot_product = BA[0] * BC[0] + BA[1] * BC[1]
        magnitude_BA = math.sqrt(BA[0]**2 + BA[1]**2)
        magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)
        cos_theta = dot_product / (magnitude_BA * magnitude_BC)
        theta_radians = math.acos(cos_theta)
        theta_degrees = math.degrees(theta_radians)
        return theta_degrees

    def func_1(self, labels, coordinates, locations):
        problem_cotent = '''\\coordinate ({E}) at ({x5}, {y5});
\\coordinate ({F}) at ({x6}, {y6});
\\coordinate ({G}) at ({x7}, {y7});
{angle1}\\draw pic [draw={color}, -, angle radius=0.3cm, angle eccentricity=2, "$\scriptstyle {degree}^\circ$"] {{angle = {G}--{F}--{E}}};
{angle2}\\tkzMarkRightAngle({G},{F},{E});'''     
        i = random.randint(0, 3)
        E, F, G = labels[i], labels[(i + 1) % 4], labels[(i + 2) % 4]
        A, B, C = coordinates[i], coordinates[(i + 1) % 4], coordinates[(i + 2) % 4]
        degree = self.calculate_angle(A, B, C)
        angle1 = ''
        angle2 = '%'
        if random.uniform(0, 1) < 0.9:
            if degree < 15:
                degree = 10
            elif degree < 25:
                degree = 20
            elif degree < 35:
                degree = 30
            elif degree < 55:
                degree = 45
            elif degree < 75:
                degree = 60
            elif degree < 89:
                degree = 80
            elif degree >= 89.9 and degree <= 90.1:
                angle1 = '%'
                angle2 = ' '
                degree = 90
            elif degree < 115:
                degree = 100
            elif degree < 135:
                degree = 120
            elif degree < 145:
                degree = 135
            elif degree < 155:
                degree = 150
            elif degree < 165: 
                degree = 160
            else:
                degree = 170
        else:
            degree = int(degree)
            if degree == 90:
                angle1 = '%'
                angle2 = ' '
        if random.uniform(0, 1) < 0.05:
            if random.uniform(0, 1) < 0.8:
                degree = random.choice(['x', 'y', 'z'])
            else:
                degree = random.sample(string.ascii_lowercase, 1)[0]
        if random.uniform(0, 1) < 0.9:
            color = 'black'
        else:
            color = random.choice([
                "red", "green", "blue", "cyan", "magenta", "yellow", "orange",
                "purple", "brown", "pink", "lime", "olive", "teal", "violet", "gray"
            ]) 
        if locations[(i + 1) % 4] == 'above right':
            location = 'below left'
        elif locations[(i + 1) % 4] == 'above left':
            location = 'below right'
        elif locations[(i + 1) % 4] == 'below right':
            location = 'above left'
        elif locations[(i + 1) % 4] == 'below left':
            location = 'above right'
        return problem_cotent.format(color=color,
                                     E=E,
                                     F=F,
                                     G=G,
                                     degree=degree,
                                     x5=A[0], y5=A[1],
                                     x6=B[0], y6=B[1],
                                     x7=C[0], y7=C[1],
                                     location=location,
                                     angle1=angle1,
                                     angle2=angle2)

    # def func_2(self):
    #     return 

    def rotate_points(self, points, angle_degrees):
        if random.uniform(0, 1) > 0.5:
            angle_degrees = 0
        x_avg = sum(pt[0] for pt in points) / len(points)
        y_avg = sum(pt[1] for pt in points) / len(points)
        angle_rad = math.radians(angle_degrees)
        rotated_points = []
        for pt in points:
            x = pt[0]
            y = pt[1]
            rotated_x = (x - x_avg) * math.cos(angle_rad) - (y - y_avg) * math.sin(angle_rad) + x_avg
            rotated_y = (x - x_avg) * math.sin(angle_rad) + (y - y_avg) * math.cos(angle_rad) + y_avg
            rotated_points.append((rotated_x, rotated_y))
        return rotated_points

    def generate_content(self):
        self.shape = self.get_shape()
        colors = [self.get_color() for _ in range(5)]        
        if self.shape == 'Parallelogram':
            coordinates = self.generate_parallelogram()
        elif self.shape == 'Rectangle':
            coordinates = self.generate_rectangle()
        elif self.shape == 'Rhombus':
            coordinates = self.generate_rhombus()
        elif self.shape == 'Trapezoid':
            coordinates = self.generate_trapezoid()
        elif self.shape == 'Square':
            coordinates = self.generate_square()
        elif self.shape == 'Right Trapezoid':
            coordinates = self.generate_right_trapezoid()
        elif self.shape == 'Isosceles Trapezoid':
            coordinates = self.generate_isosceles_trapezoid()
        else:
            while True:
                coordinates = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(4)]
                hull = ConvexHull(coordinates)
                if len(hull.vertices) == 4:
                    break
        coordinates = self.rotate_points(coordinates, random.uniform(0, 360))
        x_max, x_min = max(coordinates[0][0], coordinates[1][0], coordinates[2][0], coordinates[3][0]), min(coordinates[0][0], coordinates[1][0], coordinates[2][0], coordinates[3][0])
        y_max, y_min = max(coordinates[0][1], coordinates[1][1], coordinates[2][1], coordinates[3][1]), min(coordinates[0][1], coordinates[1][1], coordinates[2][1], coordinates[3][1])
        x_b, y_b = (x_max + x_min) / 2, (y_max + y_min) / 2
        k = random.uniform(10, 15) / max(x_max - x_min, y_max - y_min)
        coordinates = [list(coord) for coord in coordinates] 
        for i in range(4):
            coordinates[i][0] -= x_b
            coordinates[i][0] *= k
            coordinates[i][1] -= y_b
            coordinates[i][1] *= k
        coordinates = [tuple(coord) for coord in coordinates]
        # x12, y12 = (coordinates[0] + coordinates[1]) / 2
        # x23, y23 = (coordinates[1] + coordinates[2]) / 2
        # x34, y34 = (coordinates[2] + coordinates[3]) / 2
        # x41, y41 = (coordinates[3] + coordinates[1]) / 2
        # mod = random.randint(1, 97)
        # todo calculate length
        # points = coordinates.copy()
        # points.append((x12, y12))
        # location12 = self.determine_fifth_point_position(points)
        # points = coordinates.copy()
        # points.append((x23, y23))
        # location23 = self.determine_fifth_point_position(points)
        # points = coordinates.copy()
        # points.append((x34, y34))
        # location34 = self.determine_fifth_point_position(points)
        # points = coordinates.copy()
        # points.append((x41, y41))
        # location41 = self.determine_fifth_point_position(points)
        self.type = random.choice(self.types)
        # length_vis = [('%' if random.uniform(0,1) < 0.5 else '') for _ in range(4)]
        if self.type == 'Angle':
            if random.uniform(0, 1) < 0.9:
                labels = ['A', 'B', 'C', 'D', 'H']
            else:
                labels = random.sample(string.ascii_uppercase, 5)
        locations = []
        for _ in range(4):
            points = coordinates.copy()
            points.append(coordinates[_])
            locations.append(self.determine_fifth_point_position(points))
        problem_content = self.funcs[self.type](labels, coordinates, locations)
        self.new_content = self.new_content.format(
            color1=colors[0],
            color2=colors[1],
            color3=colors[2],
            color4=colors[3],
            color5=colors[4],
            x1=coordinates[0][0], y1=coordinates[0][1],
            x2=coordinates[1][0], y2=coordinates[1][1],
            x3=coordinates[2][0], y3=coordinates[2][1],
            x4=coordinates[3][0], y4=coordinates[3][1],
            # x12=x12, y12=y12,
            # x23=x23, y23=y23,
            # x34=x34, y34=y34,
            # x41=x41,y41=y41,
            # location12=location12,
            # location23=location23,
            # location34=location34,
            # location41=location41,
            # length1_vis = length_vis[1],
            # length2_vis = length_vis[2],
            # length3_vis = length_vis[3],
            # length4_vis = length_vis[4],
            label1=labels[0],
            label2=labels[1],
            label3=labels[2],
            label4=labels[3],
            location1=locations[0],
            location2=locations[1],
            location3=locations[2],
            location4=locations[3]
        )
        self.json_content = self.json_content.format(
            color1=colors[0],
            color2=colors[1],
            color3=colors[2],
            color4=colors[3],
            color5=colors[4],
            x1=coordinates[0][0], y1=coordinates[0][1],
            x2=coordinates[1][0], y2=coordinates[1][1],
            x3=coordinates[2][0], y3=coordinates[2][1],
            x4=coordinates[3][0], y4=coordinates[3][1],
            label1=labels[0],
            label2=labels[1],
            label3=labels[2],
            label4=labels[3],
            location1=locations[0],
            location2=locations[1],
            location3=locations[2],
            location4=locations[3]
        )
        
        self.new_content += problem_content
        self.json_content += problem_content

    def edit_tex_file(self, new_tex_path, new_content):
        content = f'''
            \\documentclass{{standalone}}
            \\usepackage{{tikz}}
            \\usetikzlibrary{{angles, quotes}}
            \\usepackage{{tkz-euclide}}
            \\begin{{document}}
            \\begin{{tikzpicture}}
            {new_content}
            \\draw [color=white] (-10,-10) -- (10, 10)
            \\end{{tikzpicture}}
            \\end{{document}}
        '''
        with open(new_tex_path, 'w') as file:
            file.writelines(content)

    def compile_tex_to_pdf(self, tex_path):
        os.system(f'xelatex -interaction=batchmode {tex_path} > curr.log')

    def convert_pdf_to_png(self, pdf_path, output_folder, index):
        images = convert_from_path(pdf_path, dpi=256)
        images[0].save(f'{output_folder}/page_{index}.png', 'PNG')


def process_task(index):
    success = 0
    fail = 0
    image_fold = f'./quad_angle/quad_images_{index}'
    image_fold_name = f'quad_images_{index}'
    if not os.path.exists(image_fold):
        os.makedirs(image_fold)
    new_json_file = f'./quad_angle_{index}.json'

    all_list = []
    pdar = tqdm(total=1000, desc=f'Curr_process{index} is processing')
    while (success < 1000):
        if fail > 1000:
            print('program failed too many times')
            break
        new_tex_file = f'./curr_{index}.tex'
        new_pdf_file = f'./curr_{index}.pdf'

        try:
            curr_image = drawer()
            curr_image.generate_content()
            curr_image.edit_tex_file(new_tex_file, curr_image.new_content)
            curr_image.compile_tex_to_pdf(new_tex_file)
            curr_image.convert_pdf_to_png(new_pdf_file, f'{image_fold}', success)

        except SyntaxError as e:
            print(f'curr_index{index}--{success}  SyntaxError occurs')
            fail += 1
            continue
        except IndexError as ii:
            print(f'curr_index{index}--{success}  IndexError occurs')
            fail += 1
            continue
        except ValueError as v:
            print(f'curr_index{index}--{success}  ValueError occurs')
            fail += 1
            continue
        except pp.PDFPageCountError as pc:
            print(f'curr_index{index}--{success}  PDFPageCountError occurs')
            fail += 1
            continue
        except pp.PDFSyntaxError as ps:
            print(f'curr_index{index}--{success}  PDFSyntaxError occurs')
            fail += 1
            continue
        newdict = {
            'id': success,
            'image': f'quad_angle/{image_fold_name}/page_{success}.png',
            'conversations': [{
                    'from': 'human',
                    'value': '<image>\n'
                }, {
                    'from': 'gpt',
                    'value': '\\begin{{tikzpicture}}\n' + curr_image.json_content + '\n\\end{{tikzpicture}}'
                }
            ]
        }
        all_list.append(newdict)
        pdar.update(1)
        success += 1

    with open(new_json_file, 'w') as json_file:
        json.dump(all_list, json_file, indent=4)


def gen():
    curr_image = drawer()
    curr_image.generate_content()
    return curr_image.json_content


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing.')
    parser.add_argument('--curr_index', type=int, help='curr gpu id', default=0)
    args = parser.parse_args()
    main(args)
    file_paths = [f'quad_angle_{i}.json' for i in range(10)]
    merged_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)
    with open('quad_angle.json', 'w') as output_file:
        json.dump(merged_data, output_file, indent=4)
    os.system('rm -rf quad_angle_*.json')
    os.system('rm -rf curr*')