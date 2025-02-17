import numpy as np
import math
import random
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Point


def get_shape():
    return random.choices([
            'Parallelogram',    # 平行四边形
            'Rectangle',        # 矩形
            'Rhombus',          # 菱形
            'Trapezoid',        # 梯形
            'Square',           # 正方形
            'Right Trapezoid',  # 直角梯形
            'Isosceles Trapezoid',  # 等腰梯形
            'Others', 
        ] , weights=[0.8, 0, 0.1, 0, 0.1, 0, 0, 0], k=1)[0];

def get_color():
    return random.choices(['black', 'blue', 'red', 'green', 'yellow', 'orange', 'violet', 'pink', 'cyan'], weights=[0.96, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005], k=1)[0]

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def area_of_quadrilateral(A, B, C, D):
    def triangle_area(A, B, C):
        a = distance(B, C)
        b = distance(A, C)
        c = distance(A, B)
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return area
    area1 = triangle_area(A, B, C)
    area2 = triangle_area(C, D, A)
    return area1 + area2

def angle_bisector(p1, p2, p3):
            p1, p2, p3 = map(np.array, [p1, p2, p3])
            v1 = p1 - p2
            v2 = p3 - p2
            bisector_direction = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
            bisector_direction /= np.linalg.norm(bisector_direction)
            bisector_point = p2
            return bisector_point, bisector_direction
        
def incenter(A, B, C, D):
    A, B, C, D = map(np.array, [A, B, C, D])
    AB = distance(A, B)
    BC = distance(B, C)
    CD = distance(C, D)
    DA = distance(D, A)
    s = (AB + BC + CD + DA) / 2
    area = area_of_quadrilateral(A, B, C, D)
    r = area / s
    
    def angle_bisectors_intersection(A, B, C, D):
        bisector1_point, bisector1_dir = angle_bisector(A, B, C)
        bisector2_point, bisector2_dir = angle_bisector(B, C, D)

        def line_from_point_and_direction(point, direction):
            a, b = direction
            c = -b * point[0] + a * point[1]
            return b, -a, c

        def intersection_of_lines(l1, l2):
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            A = np.array([[a1, b1], [a2, b2]])
            B = np.array([-c1, -c2])
            try:
                intersection_point = np.linalg.solve(A, B)
                return intersection_point
            except np.linalg.LinAlgError:
                return "The lines are parallel or coincident and do not intersect."
            
        l1 = line_from_point_and_direction(bisector1_point, bisector1_dir)
        l2 = line_from_point_and_direction(bisector2_point, bisector2_dir)
        return intersection_of_lines(l1, l2)

    center = angle_bisectors_intersection(A, B, C, D)
    return center, r

def generate_quadrilateral_with_incircle():
        while True:
            a = np.random.uniform(2, 5)
            b = np.random.uniform(5, 10)
            c = np.sqrt(a**2 + b**2)
            f1 = np.array([c, 0])
            f2 = np.array([-c, 0])
            x = random.uniform(2, 5)
            y = random.uniform(2, 5)
            point1 = np.array([a * np.cosh(np.pi/x), b * np.sinh(np.pi/x)])
            point2 = np.array([a * np.cosh(-np.pi/y), b * np.sinh(-np.pi/y)])
            points =  np.array([f1, f2, point1, point2])
            hull = ConvexHull(points)
            if len(hull.vertices) == 4:
                return sort_points_clockwise(points)

def sort_points_clockwise(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    return sorted_points

def generate_parallelogram():
    x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
    x2, y2 = x1 + random.uniform(3, 10), y1
    x3, y3 = x1 + random.uniform(3, 5) if random.uniform(0, 1) < 0.5 else x1 - random.uniform(3, 5), y1 + random.uniform(3, 10)
    x4 = x1 + x3 - x2
    y4 = y1 + y3 - y2
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def generate_rectangle():
    x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
    width = random.uniform(3, 6)
    height = random.uniform(7, 10)
    if random.uniform(0, 1) > 0.5:
        width, height = height, width
    x2, y2 = x1 + width, y1
    x3, y3 = x1 + width, y1 + height
    x4, y4 = x1, y1 + height
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def generate_rhombus():
    x0, y0 = random.uniform(-10, 10), random.uniform(-10, 10)
    width = random.uniform(3, 5)
    height = random.uniform(7, 10)
    x1, y1 = x0 - width / 2, y0
    x2, y2 = x0, y0 - height / 2
    x3, y3 = x0 + width / 2, y0
    x4, y4 = x0, y0 + height / 2
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def generate_trapezoid():
    x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
    base_width = random.uniform(3, 10)
    top_width = random.uniform(3, 10)
    height = random.uniform(3, 10)
    x2, y2 = x1 + base_width, y1
    x3, y3 = random.uniform(-10, 10), y1 + height
    x4, y4 = x3 - top_width, y1 + height
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def generate_square():
    x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
    size = random.uniform(3, 10)
    x2, y2 = x1 + size, y1
    x3, y3 = x1 + size, y1 + size
    x4, y4 = x1, y1 + size
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def generate_right_trapezoid():
    x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
    base_width = random.uniform(3, 10)
    top_width = random.uniform(3, 10)
    height = random.uniform(3, 10)
    x2, y2 = x1 + base_width, y1
    x3, y3 = x1 + top_width, y1 + height
    x4, y4 = x1, y1 + height
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def generate_isosceles_trapezoid():
    x1, y1 = random.uniform(-10, 10), random.uniform(-10, 10)
    base_width = random.uniform(3, 10)
    height = random.uniform(3, 10)
    top_width = random.uniform(1, base_width)
    x2, y2 = x1 + base_width, y1
    x3, y3 = x1 + base_width - (base_width - top_width) / 2, y1 + height
    x4, y4 = x1 + (base_width - top_width) / 2, y1 + height
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def projection_point(A, B, C):
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

def determine_fifth_point_position(points):
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

def is_close(point1, point2, tol=1e-9):
    return point1.distance(point2) < tol

def find_intersections_excluding_endpoints(segments, tol=1e-9):
    intersections = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            line1 = LineString([(segments[i][0], segments[i][1]), (segments[i][2], segments[i][3])])
            line2 = LineString([(segments[j][0], segments[j][1]), (segments[j][2], segments[j][3])])
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                if intersection.geom_type == 'Point':
                    point = Point(intersection.x, intersection.y)
                    if not (
                        is_close(point, Point(segments[i][0], segments[i][1]), tol) or
                        is_close(point, Point(segments[i][2], segments[i][3]), tol) or
                        is_close(point, Point(segments[j][0], segments[j][1]), tol) or
                        is_close(point, Point(segments[j][2], segments[j][3]), tol)
                    ):
                        if all(not is_close(point, existing_point, tol) for existing_point, _, _ in intersections):
                            intersections.append((point, line1, line2))
    return intersections

def get_angle(line):
    x1, y1, x2, y2 = line.xy[0][0], line.xy[1][0], line.xy[0][1], line.xy[1][1]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

def determine_position_by_angle(line1, line2):
    angle1 = get_angle(line1)
    angle2 = get_angle(line2)
    if angle1 < 0:
        angle1 += 180
    if angle2 < 0:
        angle2 += 180
    if all((angle > 22.5 and angle < 157.5) for angle in [angle1, angle2]):
        return "right"
    if all((angle < 22.5 or angle > 67.5) for angle in [angle1, angle2]):
        return "above right"
    if all((angle < 67.5 or angle > 112.5) for angle in [angle1, angle2]):
        return "above"
    if all((angle < 112.5 or angle > 157.5) for angle in [angle1, angle2]):
        return "above left"
    return "above right"

def rotate_points(points, angle_degrees):
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

def determine_position_by_line(line):
        angle = get_angle(line)
        if angle >= -22.5 and angle <= 22.5:
            return "right"
        if angle > 22.5 and angle <= 67.5:
            return "above right"
        if angle > 67.5 and angle <= 112.5:
            return "above"
        if angle > 112.5 and angle <= 157.5:
            return "above left"
        if angle > 157.5 or angle <= -157.5:
            return "left"
        if angle > -157.5 and angle <= -112.5:
            return "below left"
        if angle > -112.5 and angle <= -67.5:
            return "below"
        if angle > -67.5 and angle <= -22.5:
            return "below right"
        return "above right"


def get_locations(A, B, C, D):
    locations = []
    for _ in range(4):
        points = [A, B, C, D]
        points.append(points[_])
        locations.append(determine_fifth_point_position(points))
    return locations

def get_location(A, B, C):
    BA = (A[0] - B[0], A[1]- B[1])
    BC = (C[0] - B[0], C[1]- B[1])
    D = BA / np.linalg.norm(BA) + BC / np.linalg.norm(BC)
    return determine_position_by_line(LineString([(0, 0), D]))

def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (intersect_x, intersect_y)
    return None
