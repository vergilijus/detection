import logging
from itertools import repeat

import cv2 as cv
import webcolors

logger = logging.getLogger(__name__)

STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def get_box_points(box):
    p1 = tuple(int(round(x)) for x in box[:2])
    p2 = tuple(int(round(x)) for x in box[2:])
    return p1, p2


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index


def draw_results(img, boxes, classes=None, scores=None, labels=None, draw_classes=False):
    color_list = standard_to_bgr(STANDARD_COLORS)
    classes = classes if classes is not None else repeat(None)
    scores = scores if scores is not None else repeat(None)
    labels = labels if labels is not None else repeat(None)
    for box, class_id, score, label in zip(boxes, classes, scores, labels):
        color = color_list[class_id % len(color_list)] if class_id else (0, 255, 0)
        if not draw_classes:
            class_id = None
        draw_result(img, box, class_id, score, label, color=color)


def draw_result(img, box, class_id=None, score=None, label=None,
                color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    p1, p2 = get_box_points(box)
    cv.rectangle(img, p1, p2, color, thickness=tl)

    text = ''
    if class_id:
        text = f'{class_id} '
    if label:
        text += f'{label} '
    if score:
        text += f'{score:.0%}'

    tf = max(tl - 2, 1)  # font thickness
    text_width, text_height = cv.getTextSize(text, 0, fontScale=float(tl) / 3, thickness=tf)[0]
    text_width += 2
    text_height += 2
    x1, y1 = p1
    y1 = y1 if y1 - text_height > 0 else abs(y1 - text_height)
    p2 = x1 + text_width + 5, y1 - text_height - 3
    cv.rectangle(img, (x1, y1), p2, color, -1)  # filled
    cv.putText(img, text, (x1, y1 - 4), cv.FONT_HERSHEY_SIMPLEX, float(tl) / 3, [0, 0, 0],
               thickness=tf, lineType=cv.LINE_AA)
