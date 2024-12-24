import glob
import json
from tqdm import tqdm
import math
import random
import itertools
json_files = glob.glob('SP-1/train_sp1/sub2/jsons/*.json')

print(len(json_files))

# step is the perceptual ruler length
def get_points_on_line(x0, y0, x1, y1, step=4):    
    length = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    if length <= step:
        return [(round(x0, 2), round(y0, 2)), (round(x1, 2), round(y1, 2))]

    if length != 0:
        unit_x = (x1 - x0) / length
        unit_y = (y1 - y0) / length
    else:
        return [(x0, y0)]  

    num_points = int(length / step) + 1

    points = []
    for i in range(num_points):
        t = i * step
        if t > length:
            break
        # x = x0 + t * unit_x
        # y = y0 + t * unit_y

        if i!=0:
            offset = random.uniform(-0.5*((length/10) + (length/10)), 0.5*((length/10) + (length/10)))
            x = x0 + t * unit_x + offset
            y = y0 + t * unit_y + offset
        else:
            x = x0 + t * unit_x
            y = y0 + t * unit_y


        points.append((round(x, 2), round(y, 2)))

    if points[-1] != (x1, y1):
        points.append((x1, y1))

    return points




def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# def tikz2code(item):
#     tikz_template = """\\documentclass{{standalone}}
#     \\usepackage{{tikz}}
#     \\usetikzlibrary{{angles, quotes}}
#     \\usepackage{{tkz-euclide}}
#     \\begin{{document}}
#     \\begin{{tikzpicture}}
#     \\clip (-10, -10) rectangle (10, 10);

#     {tikz}
#     \\end{{tikzpicture}}
#     \\end{{document}}
#     """

#     tikz = ""

#     point_template = "\\node at ({x0:.2f}, {y0:.2f}) [{direction}, font=\\fontsize{{25}}{{7}}\\selectfont] {{{text}}};\n"
#     line_template = "\\draw[line width={width:.2f}pt{dash}] ({x1:.2f}, {y1:.2f}) -- ({x2:.2f}, {y2:.2f});\n"
#     circle_template = "\\draw[line width={width:.2f}pt] ({x0:.2f}, {y0:.2f}) circle ({r:.2f});\n"
#     arc_template = "\\draw[line width={width:.2f}] ({x0:.2f}, {y0:.2f}) arc[start angle={start:.2f}, end angle={end:.2f}, radius={r:.2f}];\n"
#     ellipse_template = "\\draw[line width={line_width:.2f}, shift={{({x0:.2f}, {y0:.2f})}}, rotate={theta:.2f}] (0, 0) ellipse ({width:.2f} and {height:.2f});\n"

#     if 'Line' in item['tikz']:
#         for line in item['tikz']['Line']:
#             tikz += line_template.format(width=line[1], dash="" if line[2] == '-' else ", dashed", x1=line[0][0], y1=line[0][1], x2=line[0][2], y2=line[0][3])
#             if len(line) == 4:
#                 direction = ""
#                 if line[3][0] >= 0 and line[3][1] >= 0:
#                     direction = "above right"
#                 elif line[3][0] < 0 and line[3][1] >= 0:
#                     direction = "above left"
#                 elif line[3][0] < 0 and line[3][1] < 0:
#                     direction = "below left"
#                 else:
#                     direction = "below right"
#                 tikz += point_template.format(x0=line[3][0], y0=line[3][1], direction=direction, text=line[3][2])
#     if 'Circle' in item['tikz']:
#         for circle in item['tikz']['Circle']:
#             tikz += circle_template.format(width=circle[1], x0=circle[0][0], y0=circle[0][1], r=circle[0][2])
#             if len(circle) == 4:
#                 direction = ""
#                 if circle[3][0] >= 0 and circle[3][1] >= 0:
#                     direction = "above right"
#                 elif circle[3][0] < 0 and circle[3][1] >= 0:
#                     direction = "above left"
#                 elif circle[3][0] < 0 and circle[3][1] < 0:
#                     direction = "below left"
#                 else:
#                     direction = "below right"
#                 tikz += point_template.format(x0=circle[3][0], y0=circle[3][1], direction=direction, text=circle[3][2])
#     if 'Arc' in item['tikz']:
#         for arc in item['tikz']['Arc']:
#             tikz += arc_template.format(width=arc[1], x0=arc[0][0] + arc[0][2] / 2 * math.cos(arc[0][3] * math.pi / 180), y0=arc[0][1] + arc[0][2] / 2 * math.sin(arc[0][3] * math.pi / 180), start=arc[0][3], end=arc[0][4], r=arc[0][2] / 2)
#             if len(arc) == 4:
#                 direction = ""
#                 if arc[3][0] >= 0 and arc[3][1] >= 0:
#                     direction = "above right"
#                 elif arc[3][0] < 0 and arc[3][1] >= 0:
#                     direction = "above left"
#                 elif arc[3][0] < 0 and arc[3][1] < 0:
#                     direction = "below left"
#                 else:
#                     direction = "below right"
#                 tikz += point_template.format(x0=arc[3][0], y0=arc[3][1], direction=direction, text=arc[3][2])
#     if 'Ellipse' in item['tikz']:
#         for ellipse in item['tikz']['Ellipse']:
#             tikz += ellipse_template.format(line_width=ellipse[1], x0=ellipse[0][0], y0=ellipse[0][1], theta=ellipse[0][4], width=ellipse[0][2] / 2, height=ellipse[0][3] / 2)

#     return tikz_template.format(tikz=tikz)



alist = []
for jf in tqdm(json_files):
    file = json.load(open(jf, encoding = 'utf-8'))[0]

    lines = file['Line']
    # circles = file['Circle']
    
    s = 'Line:\n'
    points_list = []
    # random.shuffle(lines)
    # print(lines)
    # exit()
    line_new = []
    for idx, line in enumerate(lines):
        # print(line[0])
        # exit()
        l = line
        x0, y0, x1, y1 = round(l[0], 2), round(l[1], 2), round(l[2], 2) , round(l[3], 2)
        sub_points = get_points_on_line(x0, y0, x1, y1)


        p0 = (x0, y0)
        p1 = (x1, y1)


        lineloc_x, lineloc_y  = round((x0 + x1)/2, 2), round((y0 + y1)/2, 2)

        ll = calculate_distance(p0, p1)


        # ss = str((lineloc_x, lineloc_y)) + ': ' + str(p0) + ' -- ' + str(p1)
        lineloc_x_0, lineloc_y_0 = round((x0 + lineloc_x)/2, 2), round((y0 + lineloc_y)/2, 2)
        lineloc_x_1, lineloc_y_1 = round((x1 + lineloc_x) / 2, 2), round((y1 + lineloc_y) / 2, 2)
        p_c = (lineloc_x, lineloc_y)
        p_a = (lineloc_x_0, lineloc_y_0)
        p_b = (lineloc_x_1, lineloc_y_1)
        points_list.append(p0)
        points_list.append(p1)

        l = [p0, p1]
        line_new.append(l)
        # ss = str(p0) +  ' -- ' + str(p1)
        # ss = str(p_c) +   ': ' + str(p0) + ' -- ' + str(p1)
        # if ll > 8:
        #     ss = str(p0) + ' -- ' + str(p_c) + ' -- ' + str(p1)
        # else:
        # ss = str(p0) + ' -- ' + str(p1)
        # ss = str(p0) + ' -- ' + str(p_a) + ' -- ' + str(p_b) + ' -- ' + str(p1)
        # ss = str(p0) + ' -- ' + str(p_c) + ' -- ' + str(p1)
        # else:
        #     ss = str(p0) + ' *----* ' + str(p1)
        ss = ''
        for idx, sub_p in enumerate(sub_points):
            if idx < len(sub_points) - 1:
                ss += str(sub_p) + ' -- '
            else:
                ss += str(sub_p)

        s += ss +'\n'
    




    p_list = list((sorted(set(points_list))))




    if p_list:
        s_p = 'Points: '
        for p in p_list:
            s_p += str(p) + '; '

        if s_p:
            s_p = s_p[:-1] + '\n'
    # s2 = 'Circle:\n'
    s2 = ''
    if 'Circle' in file.keys():
        if file['Circle']:
            s2 = 'Circle:\n'
            for idx, circle in enumerate(file['Circle']):
                # c = cicle
                # print(c)
                xc, yc, r = round(circle[0], 2), round(circle[1], 2), round(circle[2], 2)
                c = (xc, yc, r)
                s2 += str(c) + '\n'

    s3 = ''
    if 'Arc' in file.keys():
        s3 = 'Arc:\n'
        for idx, arc in enumerate(file['Arc']):
            x, y ,r, theta1, theta2 = round(arc[0], 2), round(arc[1], 2), round(arc[2], 2), round(arc[3], 2), round(arc[4], 2)
            a = (x, y ,r, theta1, theta2)
            s3 += str(idx) + ': ' + str(a) + '\n'
    # print(s)
    s4 = ''
    if 'Ellipse' in file.keys():
        s4 = 'Ellipse:\n'
        for idx, elli in enumerate(file['Ellipse']):
            center_x, center_y, width, height, angle = round(elli[0], 2), round(elli[1], 2), round(elli[2], 2), round(elli[3], 2), round(elli[4], 2)
            e = (center_x, center_y, width, height, angle)
            s4 += str(e) + '\n'



    sss = s + s2

    # sss2 = str(score)

    human_1 = '<image>\n'



    adict = {}
    img_name = jf.split('/')[-1].replace('json', 'png')
    adict['image'] = img_name
    h_dict = {}
    # if flag:
    h_dict['from'] = 'human'
    h_dict['value'] = human_1
    g_dict = {}
    g_dict['from'] = 'gpt'
    g_dict['value'] = sss





    adict['conversations'] = [h_dict, g_dict]
    alist.append(adict)


filename = '/data/jihe_render/jihe_slow_4ruler2.json'
with open(filename, 'w', encoding="utf-8") as file_obj:
    json.dump(alist, file_obj, ensure_ascii=False, indent=1)
