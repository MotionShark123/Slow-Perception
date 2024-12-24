import json
import numpy as np
import re

def sort_list_by_jpg(data):
    # 定义一个函数来提取jpg文件名中的数字
    def extract_number(item):
        filename = list(item.keys())[0]  # 获取字典的第一个（也是唯一的）键
        return int(re.search(r'\d+', filename).group())  # 提取文件名中的数字

    # 使用提取的数字作为排序键
    return sorted(data, key=extract_number)



def calculate_length(seg):
    x1, y1 = seg[0]
    x2, y2 = seg[1]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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

def calculate_f_score(pred_lines, gt_lines, iou_threshold):
    def calculate_iou(line1, line2):
        # 计算水平方向的IoU
        x_min1, x_max1 = min(line1[0][0], line1[1][0]), max(line1[0][0], line1[1][0])
        x_min2, x_max2 = min(line2[0][0], line2[1][0]), max(line2[0][0], line2[1][0])
        x_overlap = max(0, min(x_max1, x_max2) - max(x_min1, x_min2))
        x_union = max(x_max1, x_max2) - min(x_min1, x_min2)
        iou_x = x_overlap / x_union if x_union > 0 else 0

        # 计算垂直方向的IoU
        y_min1, y_max1 = min(line1[0][1], line1[1][1]), max(line1[0][1], line1[1][1])
        y_min2, y_max2 = min(line2[0][1], line2[1][1]), max(line2[0][1], line2[1][1])
        y_overlap = max(0, min(y_max1, y_max2) - max(y_min1, y_min2))
        y_union = max(y_max1, y_max2) - min(y_min1, y_min2)
        iou_y = y_overlap / y_union if y_union > 0 else 0

        # 返回水平和垂直IoU的平均值
        return (iou_x + iou_y) / 2

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 标记已匹配的真实线段
    matched_gt = set()

    # new_lines = merge_lines(pred[key])

    # print(list(new_lines))
    #
    # exit()

    for pred_line in pred_lines:
        best_iou = 0
        best_gt_idx = -1

        for i, gt_line in enumerate(gt_lines):
            if i in matched_gt:
                continue
            iou = calculate_iou(pred_line, gt_line)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou > iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1

    false_negatives = len(gt_lines) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f_score, precision, recall


def split_short_long(predictions, ground_truths, length_threshold):
    short_predictions = [pred for pred in predictions if calculate_length(pred) < length_threshold]
    long_predictions = [pred for pred in predictions if calculate_length(pred) >= length_threshold]

    short_ground_truths = [gt for gt in ground_truths if calculate_length(gt) < length_threshold]
    long_ground_truths = [gt for gt in ground_truths if calculate_length(gt) >= length_threshold]

    return short_predictions, short_ground_truths, long_predictions, long_ground_truths




def print_table(f1, f1_s, f1_l, p, p_s, p_l, r, r_s, r_l):
    # 定义列宽和表格宽度
    col_width = 10
    table_width = col_width * 4 + 5

    # 打印表头
    print("+" + "-" * table_width + "+")
    print(f"|{'Metric':^{col_width}}|{'All':^{col_width}}|{'Short':^{col_width}}|{'Long':^{col_width}}|")
    print("+" + "-" * table_width + "+")

    # 打印数据行
    print(f"|{'F1-score':^{col_width}}|{f1:^{col_width}.4f}|{f1_s:^{col_width}.4f}|{f1_l:^{col_width}.4f}|")
    print(f"|{'Precision':^{col_width}}|{p:^{col_width}.4f}|{p_s:^{col_width}.4f}|{p_l:^{col_width}.4f}|")
    print(f"|{'Recall':^{col_width}}|{r:^{col_width}.4f}|{r_s:^{col_width}.4f}|{r_l:^{col_width}.4f}|")

    # 打印表尾
    print("+" + "-" * table_width + "+")






if __name__ == "__main__" :

    # sorted_data = sort_list_by_jpg(data)
    gts = sort_list_by_jpg(json.load(open('SP-1/benchmarks/gt_120.json')))

    preds = sort_list_by_jpg(json.load(open('results/slow_perception.json')))

    print(len(gts))
    print(len(preds))

    f1_list = []
    p_list = []
    r_list = []

    f1_short_list = []
    p_short_list = []
    r_short_list = []

    f1_long_list = []
    p_long_list = []
    r_long_list = []

    match = 0

    print('===================IoU-0.75====================')

    for gt, pred in zip(gts, preds):

        # print(gt.keys())
        # exit()
        # if gt.keys() == pred.keys():
        match += 1
        key = list(gt.keys())[0]

        # print(pred[key])

        new_lines = merge_lines(pred[key])


        f_score, precision, recall = calculate_f_score(new_lines, gt[key], iou_threshold = 0.75)

        short_predictions, short_ground_truths, long_predictions, long_ground_truths = split_short_long(new_lines, gt[key], 8)

        f_short_score, short_precision, short_recall = calculate_f_score(short_predictions, short_ground_truths, iou_threshold = 0.75)
        f_long_score, long_precision, long_recall = calculate_f_score(long_predictions, long_ground_truths, iou_threshold = 0.75)

        f1_list.append(f_score)
        p_list.append(precision)
        r_list.append(recall)

        f1_short_list.append(f_short_score)
        p_short_list.append(short_precision)
        r_short_list.append(short_recall)

        f1_long_list.append(f_long_score)
        p_long_list.append(long_precision)
        r_long_list.append(long_recall)


    f1 = sum(f1_list) / len(f1_list)
    p = sum(p_list) / len(p_list)
    r = sum(r_list) / len(r_list)

    f1_short = sum(f1_short_list) / len(f1_short_list)
    p_short = sum(p_short_list) / len(p_short_list)
    r_short = sum(r_short_list) / len(r_short_list)


    f1_long = sum(f1_long_list) / len(f1_long_list)
    p_long = sum(p_long_list) / len(p_long_list)
    r_long = sum(r_long_list) / len(r_long_list)

    print_table(f1, f1_short, f1_long, p, p_short, p_long, r, r_short, r_long)

    print('Matching quantity: ', match)


    # print('===================IoU-0.90====================')
    #
    #
    # f1_list = []
    # p_list = []
    # r_list = []
    #
    # f1_short_list = []
    # p_short_list = []
    # r_short_list = []
    #
    # f1_long_list = []
    # p_long_list = []
    # r_long_list = []
    #
    #
    # for gt, pred in zip(gts, preds):
    #
    #     if gt.keys() == pred.keys():
    #         key = list(gt.keys())[0]
    #
    #         # print(pred[key])
    #
    #         new_lines = merge_lines(pred[key])
    #
    #         f_score, precision, recall = calculate_f_score(new_lines, gt[key], iou_threshold = 0.9)
    #
    #         short_predictions, short_ground_truths, long_predictions, long_ground_truths = split_short_long(new_lines, gt[key], 8)
    #
    #         f_short_score, short_precision, short_recall = calculate_f_score(short_predictions, short_ground_truths, iou_threshold = 0.9)
    #         f_long_score, long_precision, long_recall = calculate_f_score(long_predictions, long_ground_truths, iou_threshold = 0.9)
    #
    #         f1_list.append(f_score)
    #         p_list.append(precision)
    #         r_list.append(recall)
    #
    #         f1_short_list.append(f_short_score)
    #         p_short_list.append(short_precision)
    #         r_short_list.append(short_recall)
    #
    #         f1_long_list.append(f_long_score)
    #         p_long_list.append(long_precision)
    #         r_long_list.append(long_recall)
    #
    # f1 = sum(f1_list) / len(f1_list)
    # p = sum(p_list) / len(p_list)
    # r = sum(r_list) / len(r_list)
    #
    # f1_short = sum(f1_short_list) / len(f1_short_list)
    # p_short = sum(p_short_list) / len(p_short_list)
    # r_short = sum(r_short_list) / len(r_short_list)
    #
    # f1_long = sum(f1_long_list) / len(f1_long_list)
    # p_long = sum(p_long_list) / len(p_long_list)
    # r_long = sum(r_long_list) / len(r_long_list)
    #
    # print_table(f1, f1_short, f1_long, p, p_short, p_long, r, r_short, r_long)