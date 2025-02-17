from quad_angle import gen as gen1
from quad_circle1 import gen as gen2
from quad_circle2 import gen as gen3
from quad_circle3 import gen as gen4
from quad_circle4 import gen as gen5
from quad_line1 import gen as gen6
from quad_line2 import gen as gen7
from quad_line34 import gen as gen8
from quad_p1 import gen as gen9
from quad_point import gen as gen10
from quad_point2 import gen as gen11
from quad_point3 import gen as gen12
from quad_pp import gen as gen13
from quad_quad import gen as gen14
from quad_tri import gen as gen15
from quad_tri2 import gen as gen16
import re
import random
import matplotlib.pyplot as plt
import json
from joblib import Parallel, delayed
from tqdm import tqdm
from adjustText import adjust_text
import matplotlib
from typing import List, Tuple

def get_slope(p1, p2):
    """计算两个点之间的斜率。"""
    if p1[0] == p2[0]:  # 处理垂直线
        return float('inf')
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def are_equal(a, b, tolerance=1e-4):
    """检查两个浮点数是否相等，考虑容差。"""
    return abs(a - b) < tolerance

def merge_lines(segments):
    """合并相同直线的线段，取最远的两个端点。"""
    lines = {}

    for (start, end) in segments:
        # 计算斜率
        slope = get_slope(start, end)
        
        # 计算直线的截距 (y = mx + b)，使用点斜式
        intercept = start[1] - slope * start[0] if slope != float('inf') else start[0]

        # 检查现有线段的斜率和截距是否与当前线段相同
        line_key = None
        for key in lines.keys():
            existing_slope, existing_intercept = key
            if are_equal(slope, existing_slope) and are_equal(intercept, existing_intercept):
                line_key = key
                break

        if line_key is None:
            lines[(slope, intercept)] = (start, end)
        else:
            # 更新最远的两个端点
            current_start, current_end = lines[line_key]
            new_start = min(current_start, start, end)
            new_end = max(current_end, start, end)
            lines[line_key] = (new_start, new_end)

    return lines.values()

def process_value(value):
    value = re.sub(r'%.*?\n', '', value)
    pattern = r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)'
    # Find all coordinate pairs in the 'value' field
    matches = re.findall(pattern, value)
    
    # Deduplicate coordinate pairs and join them with commas
    unique_coordinates = list(set(matches))  # Convert to set to remove duplicates, then back to list
    formatted_value = '\n'.join([f'({coord[0]},{coord[1]})' for coord in unique_coordinates])
    
    # Update the 'value' field in the JSON object
    Points = formatted_value + '\n'
    # Process each conversation entry under "gpt"
    pattern = r'\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)\s*--\s*\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)\s*--\s*\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)\s*--\s*\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)\s*--\s*cycle'
    matches = re.findall(pattern, value)
    remaining_parts = re.split(pattern, value)
    remaining_str = "".join(remaining_parts)
    pattern = r'\\fill\[[^\]]+\] [^\n]+'
    remaining_str = re.sub(pattern, '', remaining_str)
    Relation = "" 
    if matches: 
        formatted_value = '\n'.join(matches)
        Relation = formatted_value + '\n'
    pattern = r"\(\s*-?\d+\.\d+,\s*-?\d+\.\d+\)(?:\s*--\s*\(\s*-?\d+\.\d+,\s*-?\d+\.\d+\))+"
    matches = re.findall(pattern, remaining_str)
    if matches: 
        formatted_value = '\n'.join(matches)
        Relation += formatted_value + '\n'
    pattern = r'\(-?\d+\.\d+, -?\d+\.\d+\)\s*circle\s*\(-?\d+\.\d+\)'
    matches = re.findall(pattern, remaining_str)
    if matches: 
        formatted_value = '\n'.join(matches)
        Relation += formatted_value + '\n'
    value = f"Points:{Points}\nRelation:{Relation}\n<image>\n"
    value = re.sub(r'.*?Relation:', '', value, flags=re.DOTALL)
    value = re.sub(r'<image>', '', value)

    # print(value)

    # 使用 re.sub 进行替换
    pattern = r'(\((?:[-+]?\d*\.\d+|[-+]?\d+),\s*(?:[-+]?\d*\.\d+|[-+]?\d+)\)\s*(?:--\s*\(.*?\)\s*)*--\s*cycle)'
    # 使用 re.sub 进行替换
    value = re.sub(pattern, lambda match: (
        match.group(0).replace(
            ' -- cycle', 
            ' -- ' + re.findall(r'(\((?:[-+]?\d*\.\d+|[-+]?\d+),\s*(?:[-+]?\d*\.\d+|[-+]?\d+)\))', match.group(0))[0]
        )
    ), value)

    # print(value)

    output_lines = []
    pattern = r'\(([^)]+)\)\s*--\s*\(([^)]+)\)'
    for i in range(len(value)):
        matches = re.findall(pattern, value[i:])

        output_lines.extend([f'({m[0]}) -- ({m[1]})' for m in matches])
    # 去重并排序
    unique_lines = sorted(set(output_lines))
    Line = '\n'.join(unique_lines)

    pattern = r'\(([^)]+)\)\s*--\s*\(([^)]+)\)'
    matches = re.findall(pattern, Line)

    Line = [ [tuple(map(float, match[0].split(','))), tuple(map(float, match[1].split(',')))] for match in matches ]
    # print('1: ', Line)
    sorted_data = [tuple(sorted(pair)) for pair in Line]
    # print('2: ', sorted_data)
    Line = sorted(set(sorted_data))
    # print('3: ', Line)
    merged_lines = merge_lines(Line)
    Line = '\n'.join(f"({start[0]}, {start[1]}) -- ({end[0]}, {end[1]})" for start, end in merged_lines)

    # print(Line)

    pattern = r'\(([-+]?\d*\.\d+|[-+]?\d+),\s*([-+]?\d*\.\d+|[-+]?\d+)\)\s*circle\s*\(([-+]?\d*\.\d+|[-+]?\d+)\)'
    # 使用 re.findall 提取所有匹配的内容
    matches = re.findall(pattern, value)
    # 将提取的结果格式化为 (x, y, r)
    Circles = [f"({x}, {y}, {r})" for x, y, r in matches]
    # 用换行符连接
    Circle = '\n'.join(Circles)

    # print(value)

    pattern = r'\(([^,]+),([^)]+)\)\s*arc\[start angle=([^,]+),end angle=([^,]+),radius=([^\]]+)\]'
    # 匹配并转换为 (x, y, r, a, b) 格式
    matches = re.findall(pattern, value)
    # 转换格式
    output = [(float(x), float(y), float(r), float(a), float(b)) for x, y, r, a, b in matches]
    Arc = '\n'.join(f"({x}, {y}, {r}, {a}, {b})" for x, y, r, a, b in output)

    # print(Arc)

    # 将新值赋给 conversation
    value = f"Line:{Line}\nCircle:{Circle}\nArc:{Arc}\n"
    return value

# Function to check if any number is out of range (-10, 10)
def is_out_of_range_1(numbers):
    return any(num < -10 or num > 10 for num in numbers)

# Function to process numbers and divide by 2
def process_number_1(n, divide_all=False):
    if divide_all:
        return n / 2
    return n

# Function to check if any number is out of range (-10, 10)
def is_out_of_range_2(numbers):
    return any(num < -20 or num > 20 for num in numbers)

# Function to process numbers and divide by 2
def process_number_2(n, divide_all=False):
    if divide_all:
        return n / 4
    return n




# Function to process a line of text and return the processed version
def process_line(line, divide_all=False):
    # divide_all = True
    # Find all the numbers in the line
    # while divide_all:
    pattern = r"(-?\d+\.\d+)"
    numbers = re.findall(pattern, line)
    
    # Convert the numbers to floats
    numbers = [float(num) for num in numbers]
    
    # processed_numbers = [str(process_number_1(num, divide_all)) for num in numbers]
    # # Check if any number is out of range (-10, 10)
    # if is_out_of_range_2(numbers):
    #     # print(2)
    #     divide_all = True
    #     processed_numbers = [str(process_number_2(num, divide_all)) for num in numbers]
    # elif is_out_of_range_1(numbers):
    #     divide_all = True
    #     processed_numbers = [str(process_number_1(num, divide_all)) for num in numbers]
    # # Process each number (divide by 2 if required)
    # processed_numbers = [float(num) for num in processed_numbers]
    # if all(num > -4 and num < 4 for num in processed_numbers) and random.uniform(0, 1) > 0.2:
    #     processed_numbers = [str(num * 2) for num in processed_numbers]


    if len(numbers) % 2 == 0:
        x = [numbers[i] for i in range(0, len(numbers), 2)]
        y = [numbers[i] for i in range(1, len(numbers), 2)]    
    else:
        x = [numbers[i] for i in range(0, len(numbers) - 1, 2)]
        y = [numbers[i] for i in range(1, len(numbers) - 1, 2)]
    x_max, x_min, y_max, y_min = max(x), min(x), max(y), min(y)
    x_b = (x_max + x_min) / 2
    y_b = (y_max + y_min) / 2
    k = random.randint(14,17) / max(x_max - x_min, y_max - y_min)
    # print(k, ' ', x_max-x_min, ' ', y_max-y_min)
    processed_numbers = [num for num in numbers]
    if len(numbers) % 2 == 0:
        for i in range(0, len(numbers), 2):
            processed_numbers[i] -= x_b
            processed_numbers[i] *= k
            processed_numbers[i + 1] -= y_b 
            processed_numbers[i + 1] *= k
    else:
        for i in range(0, len(numbers) - 1, 2):
            processed_numbers[i] -= x_b
            processed_numbers[i] *= k
            processed_numbers[i + 1] -= y_b 
            processed_numbers[i + 1] *= k
        processed_numbers[len(numbers) - 1] *= k

    # print(numbers, processed_numbers)
    processed_numbers = [round(num, 2) for num in processed_numbers]
    processed_numbers = [str(num) for num in processed_numbers]
    # Replace the original numbers with processed ones
    pos = 0  # 初始化位置为字符串开头
    for original, processed in zip(numbers, processed_numbers):
        original_str = str(original) 
        match = re.search(original_str, line[pos:])  # 从当前位置开始查找
        if not match:
            break  # 如果找不到匹配项，则退出循环
        start, end = match.span()  # 获取匹配的起止位置
        line = line[:pos + start] + processed + line[pos + end:]  # 替换第一个找到的匹配项
        pos += start + len(processed)  # 更新当前位置
        
    return line


def getNodePos(text_list):
    offset2 = random.choice([0.1, 0.3, 0.4, 0.5])
    for index, text in enumerate(text_list):
        x, y = text[0], text[1]
        offset = random.choice([0.1,0.15,0.2])
        
        if x>=0 and y>=0 :
            # return 'above right'
            text_list[index][0] = text_list[index][0] + offset + offset2
            text_list[index][1] = text_list[index][1] + offset + offset2
            text_list[index].append(['center', 'center'])

        elif x<0 and y >=0 :
            # return 'above left'
            text_list[index][0] = text_list[index][0] - offset - offset2
            text_list[index][1] = text_list[index][1] + offset + offset2
            text_list[index].append(['center', 'center'])

        elif x<0 and y<0 :
            # return 'below left'
            text_list[index][0] = text_list[index][0] - offset - offset2
            text_list[index][1] = text_list[index][1] - offset - offset2
            text_list[index].append(['center', 'center'])
        else :
            # return 'below right'
            text_list[index][0] = text_list[index][0] + offset + offset2
            text_list[index][1] = text_list[index][1] - offset - offset2
            text_list[index].append(['center', 'center'])
    return text_list


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9

    def __hash__(self):
        return hash((self.x, self.y))


class Segment:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end


def orientation(p: Point, q: Point, r: Point) -> int:
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if abs(val) < 1e-9:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise


def on_segment(p: Point, q: Point, r: Point) -> bool:
    return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
            q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))


def intersect(seg1: Segment, seg2: Segment) -> Point:
    p1, q1 = seg1.start, seg1.end
    p2, q2 = seg2.start, seg2.end

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        x1, y1 = p1.x, p1.y
        x2, y2 = q1.x, q1.y
        x3, y3 = p2.x, p2.y
        x4, y4 = q2.x, q2.y

        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
            ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
            ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        intersection = Point(x, y)

        # 检查交点是否为任一线段的端点
        if intersection == p1 or intersection == q1 or intersection == p2 or intersection == q2:
            return None

        return intersection

    return None


def find_intersections(segments: List[Segment]) -> List[Point]:
    intersections = set()
    n = len(segments)

    for i in range(n):
        for j in range(i + 1, n):
            intersection = intersect(segments[i], segments[j])
            if intersection:
                intersections.add(intersection)

    return list(intersections)


# if __name__ == "__main__":
def get_lines(idx):

    gen_functions = [gen2, gen3, gen4, gen5, gen6, gen7, gen8, gen9, gen10, gen11, gen12, gen13, gen14, gen15, gen16]
    selected_gen = random.choice(gen_functions)
    # for i in tqdm(range(1000000), desc="Processing"):
    # print(process_line((process_value(selected_gen()))))
    outputs_think = process_line((process_value(selected_gen())))
    # print(outputs_think)
    outputs_think_p = outputs_think.split('Line:')[1].split('Circle:')[0]
    outputs_think_c = outputs_think.split('Line:')[1].split('Circle:')[1].split('Arc:')[0]
    
    # print(outputs_think_p)
    outputs_think_p_list = outputs_think_p.split('\n')
    outputs_think_c_list = outputs_think_c.split('\n')
        # print(outputs_think_p_list)
    # try:
    # dpi = random.choice([32, 54, 54, 72, 72, 72, 150, 150, 300, 300])
    dpi = 96
    fig, ax = plt.subplots(figsize=(2.8,2.8), dpi=96)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.axis('off')
    # plt.axis('none')
    line_list = []
    line_cache = []
    circle_list = []
    text_list = []
    line_text_cache = []
    tikz = {}
    tikz_line = []
    tikz_circle = []
    tikz_arc = []
    # linewidth = random.uniform(0.5, 8)
    linewidth = random.uniform(0.6, 0.8)
    # flag_antialiased = random.choice([0, 1])
    flag_antialiased = 0

    if flag_antialiased:
        plt.rcParams['text.antialiased'] = False

    for p in outputs_think_p_list:
        # try:
        if '--' in p:
            pp_list = p.split(' -- ')
            
            # print(pp_list)
            
            # for i in range(len(pp_list) - 1):
            pp_list[0] = pp_list[0].replace('--', '-')
            pp_list[1] = pp_list[1].replace('--', '-')
            p0 = eval(pp_list[0])
            p1 = eval(pp_list[1])
            # linewidth = random.uniform(0.5, 8)
            line_list.append([p0[0], p0[1], p1[0], p1[1]])
            line_cache = [(p0[0], p0[1], p1[0], p1[1]), linewidth]
            flag = random.randint(1,5)
            # flag2 = random.choice([0, 1])
            if flag == 4:
                if flag_antialiased:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='k', linestyle='--', linewidth=linewidth, antialiased=False)
                    # ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linestyle='--', linewidth=linewidth,
                    #         antialiased=False)
                else:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='k', linestyle='--', linewidth=linewidth)
                    # ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linestyle='--', linewidth=linewidth)
                line_cache.append('--')
            else:
                if flag_antialiased:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='k',  linewidth=linewidth, antialiased=False)
                    # ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=linewidth, antialiased=False)
                else:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='k',  linewidth=linewidth)
                    # ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=linewidth)
                line_cache.append('-')
            # ax.plot([p0[0], p1[0]], [p0[1], p1[1]])

            if flag > 2:
                texts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                text1 = random.choice(texts)
                if [p0[0], p0[1]] not in line_text_cache:
                    line_text_cache.append([p0[0], p0[1]])
                    text_list.append([p0[0], p0[1], text1])
                    line_cache.append([p0[0], p0[1], text1])

            tikz_line.append(line_cache)

    # print(text_list)

    font_path = random.choice(['/home/ucaswei/.fonts/Times_rotate.ttf', '/home/ucaswei/.fonts/times.ttf'])  # 你的字体文件路径  #/home/ucaswei/.fonts/Times_rotate.ttf   /home/ucaswei/.fonts/times.ttf'
    prop = matplotlib.font_manager.FontProperties(fname=font_path)
    footdict = {}
    footdict['family'] = 'Times New Roman'

    text_list = getNodePos(text_list)

    # fontsize = random.choice([25, 30, 35])
    fontsize = random.choice([10, 12, 9])

    texts = [ax.text(x, y, label, footdict, fontsize= fontsize, color = 'k', fontproperties=prop, ha = loc[0], va = loc[1]) for x, y, label, loc in text_list]

    adjust_text(texts)

    circle_texts = []
    circle_cache = []

    for c in outputs_think_c_list:
        if c:
            c = eval(c)
            r = float(c[2])
            center = (c[0], c[1])

            # radius = random.uniform(2, 5)

            if flag_antialiased:
                circle = plt.Circle(center, r, color='k', linewidth = linewidth, fill=False, antialiased=False)
            else:
                circle = plt.Circle(center, r, color='k', linewidth = linewidth, fill=False)

            circle_list.append([center[0], center[1], r])
            circle_cache = [(center[0], center[1], r), linewidth]

            circle_cache.append('-')

            flag6 = random.choice([0, 1])

            if flag6:
                circle_texts.append([c[0], c[1], 'O'])
                circle_cache.append([c[0], c[1], 'O'])
            ax.add_artist(circle)
            tikz_circle.append(circle_cache)

    circle_texts = getNodePos(circle_texts)

    texts4 = [ax.text(x, y, label, footdict, fontsize= fontsize, color = 'k', fontproperties=prop, ha = loc[0], va = loc[1]) for x, y, label, loc in circle_texts]

    adjust_text(texts4)




    # print(line_list)
    l_list_new = []

    save_line_list = []

    for idx2, ll in enumerate(line_list):
        # print(ll)
        p0_xold, p0_yold, p1_xold, p1_yold = ll[0], ll[1], ll[2], ll[3]
        l_list_new.append(Segment(Point(p0_xold, p0_yold), Point(p1_xold, p1_yold)))

        p0 = ax.transData.transform((ll[0], ll[1])).tolist()
        p0_x = p0[0]
        # p0_y = dpi*10 - p0[1]
        p0_y = p0[1]
        p1 = ax.transData.transform((ll[2], ll[3])).tolist()
        p1_x = p1[0]
        p1_y = p1[1]

        # p1_y = dpi*10 - p1[1]

        p0_x_new = round(p0_x / (dpi * 2.8) * 20 - 10, 2)
        p0_y_new = round(p0_y / (dpi * 2.8) * 20 - 10, 2)

        p1_x_new = round(p1_x / (dpi * 2.8) * 20 - 10, 2)
        p1_y_new = round(p1_y / (dpi * 2.8) * 20 - 10, 2)

        # tikz_line[]



        save_line_list.append([p0_x_new, p0_y_new, p1_x_new, p1_y_new])


    save_circle_list = []

    for cc in circle_list:
        c0_x, c0_y, r = cc[0], cc[1], cc[2]

        c1_x, c1_y = c0_x + r, c0_y

        c0_new = ax.transData.transform((c0_x, c0_y)).tolist()

        c1_new = ax.transData.transform((c1_x, c1_y)).tolist()

        c0_x_new = round(c0_new[0] / (dpi * 2.8) * 20 - 10, 2)
        c0_y_new = round(c0_new[1] / (dpi * 2.8) * 20 - 10, 2)

        c1_x_new = round(c1_new[0] / (dpi * 2.8) * 20 - 10, 2)

        r_new = c1_x_new - c0_x_new

        save_circle_list.append([c0_x_new, c0_y_new, r_new])

        



    # print([p0_x, p0_y, p1_x, p1_y])

    intersections = find_intersections(l_list_new)
    # print(intersections)

    inter_points = []
    inter_cache = []
    for point in intersections:
        # print(f"({point.x:.2f}, {point.y:.2f})")
        x0 = round(point.x, 2)
        y0 = round(point.y, 2)
        texts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']
        text1 = random.choice(texts)
        flag_inter = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        if flag_inter:
            if [x0,y0] not in inter_cache:
                inter_cache.append([x0,y0])
                inter_points.append([x0, y0, text1])
        # ax.text(x0, y0, 'X', footdict, fontsize=fontsize, color='k', fontproperties=prop)

    inter_points = getNodePos(inter_points)

    texts5 = [ax.text(x, y, label, footdict, fontsize= fontsize, color = 'k', fontproperties=prop, ha = loc[0], va = loc[1]) for x, y, label, loc in inter_points]

    adjust_text(texts5)



    adict = {}
    adict['Line'] = save_line_list
    if circle_list:
        adict['Circle'] = save_circle_list

    tikz['Line'] = tikz_line
    if tikz_circle:
        tikz['Circle'] = tikz_circle
    if tikz_arc:
        tikz['Arc'] = tikz_arc

    # adict['tikz'] = tikz

    # print(adict)


    # plt.savefig('/data/show/figure.png', bbox_inches='tight')
    # print(dpi)
    plt.savefig('/data/jihe_render/sft_jihe/imgs/' + str(idx) + '_det.png')
    plt.close()
    # print(adict)
    filename = '/data/jihe_render/sft_jihe/jsons/' + str(idx) + '_det.json'
    with open(filename, 'w', encoding="utf-8") as file_obj:
        json.dump([adict], file_obj, ensure_ascii=False, indent=1)



Parallel(n_jobs=-1)(delayed(get_lines)(p) for p in tqdm(range(150000)))

# get_lines(1)