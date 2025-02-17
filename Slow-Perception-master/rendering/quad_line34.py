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
import re
from shapely.geometry import LineString, Point


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
            'Line34'
        ]
        self.funcs = {
            'Line34': self.func_1
        }
        self.colors = ['black', 'blue', 'red', 'green', 'yellow', 'orange', 'violet', 'pink', 'cyan']
        self.new_content = '''\\draw [thick, {color1}] ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- ({x4},{y4}) -- cycle;
\\node [{location1}, {color2}, font=\scriptsize] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color3}, font=\scriptsize] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color4}, font=\scriptsize] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color5}, font=\scriptsize] at ({x4},{y4}) {{{label4}}};
'''
# {length1_vis}\\node at [{location12}] ({x12},{y12}) {{{label12}}};
# {length2_vis}\\node at [{location23}] ({x23},{y23}) {{{label23}}};
# {length3_vis}\\node at [{location34}] ({x34},{y34}) {{{label34}}};
# {length4_vis}\\node at [{location41}] ({x41},{y41}) {{{label41}}};
# '''
        self.json_content = '''\\draw [thick, {color1}] ({x1},{y1}) -- ({x2},{y2}) -- ({x3},{y3}) -- ({x4},{y4}) -- cycle;
\\node [{location1}, {color2}, font=\scriptsize] at ({x1},{y1}) {{{label1}}};
\\node [{location2}, {color3}, font=\scriptsize] at ({x2},{y2}) {{{label2}}};
\\node [{location3}, {color4}, font=\scriptsize] at ({x3},{y3}) {{{label3}}};
\\node [{location4}, {color5}, font=\scriptsize] at ({x4},{y4}) {{{label4}}};
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
        
    
    def is_close(self, point1, point2, tol=1e-9):
        return point1.distance(point2) < tol


    def find_intersections_excluding_endpoints(self, segments, tol=1e-9):
        intersections = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                line1 = LineString([(segments[i][0], segments[i][1]), (segments[i][2], segments[i][3])])
                line2 = LineString([(segments[j][0], segments[j][1]), (segments[j][2], segments[j][3])])
                if line1.intersects(line2):
                    intersection = line1.intersection(line2)
                    if intersection.geom_type == 'Point':
                        point = Point(intersection.x, intersection.y)
                        # 检查交点是否与任一线段的端点非常接近
                        if not (
                            self.is_close(point, Point(segments[i][0], segments[i][1]), tol) or
                            self.is_close(point, Point(segments[i][2], segments[i][3]), tol) or
                            self.is_close(point, Point(segments[j][0], segments[j][1]), tol) or
                            self.is_close(point, Point(segments[j][2], segments[j][3]), tol)
                        ):
                            if all(not self.is_close(point, existing_point, tol) for existing_point, _, _ in intersections):
                                intersections.append((point, line1, line2))
        return intersections


    def get_angle(self, line):
        x1, y1, x2, y2 = line.xy[0][0], line.xy[1][0], line.xy[0][1], line.xy[1][1]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle


    def determine_position_by_angle(self, line1, line2):
        angle1 = self.get_angle(line1)
        angle2 = self.get_angle(line2)
        if angle1 < 0:
            angle1 += 180
        if angle2 < 0:
            angle2 += 180
        # print(f'Angles: {angle1}, {angle2}')
        if all((angle > 22.5 and angle < 157.5) for angle in [angle1, angle2]):
            return "right"
        if all((angle < 22.5 or angle > 67.5) for angle in [angle1, angle2]):
            return "above right"
        if all((angle < 67.5 or angle > 112.5) for angle in [angle1, angle2]):
            return "above"
        if all((angle < 112.5 or angle > 157.5) for angle in [angle1, angle2]):
            return "above left"
        return "above right"


    def func_1(self, labels, coordinates, colors):
        problem_cotent = '''\\coordinate ({start}) at ({x5}, {y5});
\\coordinate ({middle}) at ({x6}, {y6});
\\coordinate ({end}) at ({x7}, {y7});
\\coordinate ({label1}) at ({x1}, {y1});
\\coordinate ({label2}) at ({x2}, {y2});
\\coordinate ({label3}) at ({x3}, {y3});
\\coordinate ({label4}) at ({x4}, {y4});
{vis_1}\\draw [{dashed1}, {color1}] ({x1}, {y1}) -- ({x3}, {y3});
{vis_2}\\draw [{dashed1}, {color1}] ({x2}, {y2}) -- ({x4}, {y4});
{vis_s}\\node [{location_s}, {color_s}, font=\scriptsize] at ({x5},{y5}) {{{start}}};
{vis_m}\\node [{location_m}, {color_m}, font=\scriptsize] at ({x6},{y6}) {{{middle}}};
{vis_e}\\node [{location_e}, {color_e}, font=\scriptsize] at ({x7},{y7}) {{{end}}};
\\draw [{dashed1}, {color1}] ({x5}, {y5}) -- ({x6}, {y6});
\\draw [{dashed2}, {color2}] ({x6}, {y6}) -- ({x7}, {y7});
{vis_12}\\node [{location_12}, {color_12}, font=\scriptsize] at ({x12},{y12}) {{{label12}}};
{vis_15}\\node [{location_15}, {color_15}, font=\scriptsize] at ({x15},{y15}) {{{label15}}};
{vis_16}\\node [{location_16}, {color_16}, font=\scriptsize] at ({x16},{y16}) {{{label16}}};
{vis_25}\\node [{location_25}, {color_25}, font=\scriptsize] at ({x25},{y25}) {{{label25}}};
{vis_26}\\node [{location_26}, {color_26}, font=\scriptsize] at ({x26},{y26}) {{{label26}}};
{vis_56}\\node [{location_56}, {color_56}, font=\scriptsize] at ({x56},{y56}) {{{label56}}};
'''     
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        if random.uniform(0, 1) < 0.5:
            if random.uniform(0, 1) < 0.5:
                vis_1, vis_2 = '', '%'
            else:
                vis_1, vis_2 = '%', ''
        else:
            vis_1, vis_2 = '', ''
        dashed1 = 'dashed' if random.uniform(0, 1) < 0.5 else 'thick'
        dashed2 = dashed1
        color_s, color_m, color_e = self.get_color(), self.get_color(), self.get_color()
        color_12, color_15, color_16, color_25, color_26, color_56 = self.get_color(), self.get_color(), self.get_color(), self.get_color(), self.get_color(), self.get_color()
        if random.uniform(0, 1) < 0.9:
            color1 = colors[0]
        else:
            color1 = self.get_color()
        if random.uniform(0, 1) < 0.9:
            color2 = colors[0]
        else:
            color2 = self.get_color()
        if random.uniform(0, 1) < 0.3:
            start_index = random.randint(0, 3)
            end_index = (start_index + 1) % 4
            vis_s, vis_e = '%', '%'
            start, end = labels[start_index], labels[end_index]
            (x5, y5) = coordinates[start_index]
            (x7, y7) = coordinates[end_index]
            vis_m = ''
            middle = labels[4]
            if random.uniform(0, 1) < 0.9:
                x6 = (coordinates[(start_index + 2) % 4][0] + coordinates[(start_index + 3) % 4][0]) / 2
                y6 = (coordinates[(start_index + 2) % 4][1] + coordinates[(start_index + 3) % 4][1]) / 2
            else:
                i = random.uniform(0.3, 0.7)
                x6 = i * coordinates[(start_index + 2) % 4][0] + (1 - i) * coordinates[(start_index + 3) % 4][0]
                y6 = i * coordinates[(start_index + 2) % 4][1] + (1 - i) * coordinates[(start_index + 3) % 4][1]
        elif random.uniform(0, 1) < 0.5:
            if random.uniform(0, 1) < 0.5:
                start_index = random.randint(0, 3)
                vis_s = '%'
                start = labels[start_index]
                (x5, y5) = coordinates[start_index]
                vis_m, vis_e = '', ''
                middle, end = labels[4], labels[5]
                if random.uniform(0, 1) < 0.9:
                    x6 = (coordinates[(start_index + 1) % 4][0] + coordinates[(start_index + 2) % 4][0]) / 2
                    y6 = (coordinates[(start_index + 1) % 4][1] + coordinates[(start_index + 2) % 4][1]) / 2
                else:
                    i = random.uniform(0.3, 0.7)
                    x6 = i * coordinates[(start_index + 1) % 4][0] + (1 - i) * coordinates[(start_index + 2) % 4][0]
                    y6 = i * coordinates[(start_index + 1) % 4][1] + (1 - i) * coordinates[(start_index + 2) % 4][1]
                if random.uniform(0, 1) < 0.9:
                    x7 = (coordinates[(start_index + 2) % 4][0] + coordinates[(start_index + 3) % 4][0]) / 2
                    y7 = (coordinates[(start_index + 2) % 4][1] + coordinates[(start_index + 3) % 4][1]) / 2
                else:
                    i = random.uniform(0.3, 0.7)
                    x7 = i * coordinates[(start_index + 2) % 4][0] + (1 - i) * coordinates[(start_index + 3) % 4][0]
                    y7 = i * coordinates[(start_index + 2) % 4][1] + (1 - i) * coordinates[(start_index + 3) % 4][1]
                if random.uniform(0, 1) < 0.5:
                    x6, y6, x7, y7 = x7, y7, x6, y6
            else:
                middle_index = random.randint(0, 3)
                vis_m = '%'
                middle = labels[middle_index]
                (x6, y6) = coordinates[middle_index]
                vis_s, vis_e = '', ''
                start, end = labels[4], labels[5]
                if random.uniform(0, 1) < 0.9:
                    x5 = (coordinates[(middle_index + 1) % 4][0] + coordinates[(middle_index + 2) % 4][0]) / 2
                    y5 = (coordinates[(middle_index + 1) % 4][1] + coordinates[(middle_index + 2) % 4][1]) / 2
                else:
                    i = random.uniform(0.3, 0.7)
                    x5 = i * coordinates[(middle_index + 1) % 4][0] + (1 - i) * coordinates[(middle_index + 2) % 4][0]
                    y5 = i * coordinates[(middle_index + 1) % 4][1] + (1 - i) * coordinates[(middle_index + 2) % 4][1]
                if random.uniform(0, 1) < 0.9:
                    x7 = (coordinates[(middle_index + 2) % 4][0] + coordinates[(middle_index + 3) % 4][0]) / 2
                    y7 = (coordinates[(middle_index + 2) % 4][1] + coordinates[(middle_index + 3) % 4][1]) / 2
                else:
                    i = random.uniform(0.3, 0.7)
                    x7 = i * coordinates[(middle_index + 2) % 4][0] + (1 - i) * coordinates[(middle_index + 3) % 4][0]
                    y7 = i * coordinates[(middle_index + 2) % 4][1] + (1 - i) * coordinates[(middle_index + 3) % 4][1]
        else:
            vis_s, vis_m, vis_e = '', '', ''
            start, middle, end = labels[4], labels[5], labels[6]
            start_index = random.randint(0, 3)
            if random.uniform(0, 1) < 0.9:
                x5 = (coordinates[(start_index) % 4][0] + coordinates[(start_index + 1) % 4][0]) / 2
                y5 = (coordinates[(start_index) % 4][1] + coordinates[(start_index + 1) % 4][1]) / 2
            else:
                i = random.uniform(0.3, 0.7)
                x5 = i * coordinates[(start_index) % 4][0] + (1 - i) * coordinates[(start_index + 1) % 4][0]
                y5 = i * coordinates[(start_index) % 4][1] + (1 - i) * coordinates[(start_index + 1) % 4][1]
            if random.uniform(0, 1) < 0.9:
                x6 = (coordinates[(start_index + 1) % 4][0] + coordinates[(start_index + 2) % 4][0]) / 2
                y6 = (coordinates[(start_index + 1) % 4][1] + coordinates[(start_index + 2) % 4][1]) / 2
            else:
                i = random.uniform(0.3, 0.7)
                x6 = i * coordinates[(start_index + 1) % 4][0] + (1 - i) * coordinates[(start_index + 2) % 4][0]
                y6 = i * coordinates[(start_index + 1) % 4][1] + (1 - i) * coordinates[(start_index + 2) % 4][1]
            if random.uniform(0, 1) < 0.9:
                x7 = (coordinates[(start_index + 2) % 4][0] + coordinates[(start_index + 3) % 4][0]) / 2
                y7 = (coordinates[(start_index + 2) % 4][1] + coordinates[(start_index + 3) % 4][1]) / 2
            else:
                i = random.uniform(0.3, 0.7)
                x7 = i * coordinates[(start_index + 2) % 4][0] + (1 - i) * coordinates[(start_index + 3) % 4][0]
                y7 = i * coordinates[(start_index + 2) % 4][1] + (1 - i) * coordinates[(start_index + 3) % 4][1]
            if random.uniform(0, 1) < 0.5:
                x6, y6, x7, y7 = x7, y7, x6, y6
        points = coordinates.copy()
        points.append((x5, y5))
        location_s = self.determine_fifth_point_position(points)
        points = coordinates.copy()
        points.append((x6, y6))
        location_m = self.determine_fifth_point_position(points)
        points = coordinates.copy()
        points.append((x7, y7))
        location_e = self.determine_fifth_point_position(points)


        points = []
        if vis_1 == '':
            points.append((x1, y1, x3, y3))
        if vis_2 == '':
            points.append((x2, y2, x4, y4))
        points.append((x5, y5, x6, y6))
        points.append((x6, y6, x7, y7))
        intersections = self.find_intersections_excluding_endpoints(points)
        vis_12, vis_15, vis_16, vis_25, vis_26, vis_56 = '%', '%', '%', '%', '%', '%'
        (x12, y12), (x15, y15), (x16, y16), (x25, y25), (x26, y26), (x56, y56) = (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        location_12 = 'above right'
        location_15 = 'above right'
        location_16 = 'above right'
        location_25 = 'above right'
        location_26 = 'above right'
        location_56 = 'above right'
        if len(intersections) > 0:
            vis_12 = ''
            (x12, y12) = (intersections[0][0].x, intersections[0][0].y)
            location_12 = self.determine_position_by_angle(intersections[0][1], intersections[0][2])
        if len(intersections) > 1:
            vis_15 = ''
            (x15, y15) = (intersections[1][0].x, intersections[1][0].y)
            location_15 = self.determine_position_by_angle(intersections[1][1], intersections[1][2])
        if len(intersections) > 2:
            vis_16 = ''
            (x16, y16) = (intersections[2][0].x, intersections[2][0].y)
            location_16 = self.determine_position_by_angle(intersections[2][1], intersections[2][2])
        if len(intersections) > 3:
            vis_25 = ''
            (x25, y25) = (intersections[3][0].x, intersections[3][0].y)
            location_25 = self.determine_position_by_angle(intersections[3][1], intersections[3][2])
        if len(intersections) > 4:
            vis_26 = ''
            (x26, y26) = (intersections[4][0].x, intersections[4][0].y)
            location_26 = self.determine_position_by_angle(intersections[4][1], intersections[4][2])
        if len(intersections) > 5:
            vis_56 = ''
            (x56, y56) = (intersections[5][0].x, intersections[5][0].y)
            location_56 = self.determine_position_by_angle(intersections[5][1], intersections[5][2])
        return problem_cotent.format(x5=x5, y5=y5,
                                     x6=x6, y6=y6,
                                     x7=x7, y7=y7,
                                     start=start,
                                     middle=middle,
                                     end=end,
                                     vis_s=vis_s,
                                     vis_m=vis_m,
                                     vis_e=vis_e,
                                     location_s=location_s,
                                     location_m=location_m,
                                     location_e=location_e,
                                     color_s=color_s,
                                     color_m=color_m,
                                     color_e=color_e,
                                     color1=color1,
                                     color2=color2,
                                     dashed1=dashed1,
                                     dashed2=dashed2,
                                     x1=x1, y1=y1,
                                     x2=x2, y2=y2,
                                     x3=x3, y3=y3,
                                     x4=x4, y4=y4,
                                     vis_1=vis_1,
                                     vis_2=vis_2,
                                     label1 = labels[0],
                                     label2 = labels[1],
                                     label3 = labels[2],
                                     label4 = labels[3],
                                     color_12=color_12,
                                     color_15=color_15,
                                     color_16=color_16,
                                     color_25=color_25,
                                     color_26=color_26,
                                     color_56=color_56,
                                     label12=labels[7],
                                     label15=labels[8],
                                     label16=labels[9],
                                     label25=labels[10],
                                     label26=labels[11],
                                     label56=labels[12],
                                     vis_12=vis_12,
                                     vis_15=vis_15,
                                     vis_16=vis_16,
                                     vis_25=vis_25,
                                     vis_26=vis_26,
                                     vis_56=vis_56,
                                     x12=x12, y12=y12,
                                     x15=x15, y15=y15,
                                     x16=x16, y16=y16,
                                     x25=x25, y25=y25,
                                     x26=x26, y26=y26,
                                     x56=x56, y56=y56,
                                     location_12=location_12,
                                     location_15=location_15,
                                     location_16=location_16,
                                     location_25=location_25,
                                     location_26=location_26,
                                     location_56=location_56,
                                    )

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
        if self.type == 'Line34':
            if random.uniform(0, 1) < 0.9:
                labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
            else:
                labels = random.sample(string.ascii_uppercase, 13)
        locations = []
        for _ in range(4):
            points = coordinates.copy()
            points.append(coordinates[_])
            locations.append(self.determine_fifth_point_position(points))
        problem_content = self.funcs[self.type](labels, coordinates, colors)
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
            # x41=x41, y41=y41,
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
    image_fold = f'./quad_line34/quad_images_{index}'
    image_fold_name = f'quad_images_{index}'
    if not os.path.exists(image_fold):
        os.makedirs(image_fold)
    new_json_file = f'./quad_line34_{index}.json'

    all_list = []
    pdar = tqdm(total=10, desc=f'Curr_process{index} is processing')
    while (success < 10):
        if fail > 10:
            print('program failed too many times')
            break
        new_tex_file = f'./curr_{index}.tex'
        new_pdf_file = f'./curr_{index}.pdf'

        # try:
        curr_image = drawer()
        curr_image.generate_content()
        curr_image.edit_tex_file(new_tex_file, curr_image.new_content)
        curr_image.compile_tex_to_pdf(new_tex_file)
        curr_image.convert_pdf_to_png(new_pdf_file, f'{image_fold}', success)

        # except SyntaxError as e:
        #     print(f'curr_index{index}--{success}  SyntaxError occurs')
        #     fail += 1
        #     continue
        # except IndexError as ii:
        #     print(f'curr_index{index}--{success}  IndexError occurs')
        #     fail += 1
        #     continue
        # except ValueError as v:
        #     print(f'curr_index{index}--{success}  ValueError occurs')
        #     fail += 1
        #     continue
        # except pp.PDFPageCountError as pc:
        #     print(f'curr_index{index}--{success}  PDFPageCountError occurs')
        #     fail += 1
        #     continue
        # except pp.PDFSyntaxError as ps:
        #     print(f'curr_index{index}--{success}  PDFSyntaxError occurs')
        #     fail += 1
        #     continue
        newdict = {
            'id': success,
            'image': f'quad_line34/{image_fold_name}/page_{success}.png',
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


def main(args):
    curr_index = args.curr_index
    process_num = 1
    processes = []
    for i in range(process_num): 
        p = Process(target=process_task, args=((i % process_num + curr_index * process_num),))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def process_value(value):
    return re.sub(r'%.*?\n', '', value)


def process_conversations(conversations):
    for conversation in conversations:
        if 'value' in conversation:
            conversation['value'] = process_value(conversation['value'])


def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    for item in data:
        if 'conversations' in item:
            process_conversations(item['conversations'])
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)


def gen():
    curr_image = drawer()
    curr_image.generate_content()
    return curr_image.json_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing.')
    parser.add_argument('--curr_index', type=int, help='curr gpu id', default=0)
    args = parser.parse_args()
    main(args)
    file_paths = [f'quad_line34_{i}.json' for i in range(10)]
    merged_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)
    with open('quad_line34_pre.json', 'w') as output_file:
        json.dump(merged_data, output_file, indent=4)
    os.system('rm -rf curr*')
    input_file = 'quad_line34_pre.json'
    output_file = 'quad_line34.json'
    process_json(input_file, output_file)
    os.system('rm -rf quad_line34_*.json')
