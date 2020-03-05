__author__ = "wangyingwu"
__email__ = "wangyw.zero@gmail.com"


def segment_list(list_in, num_per_part):
    """
    等分列表
    :param list_in: 要划分的列表
    :param num_per_part: 每一份的数目
    :return: 划分后的结果
    """
    def segment_list_to_generator(list_in, num_per_part):
        for i in range(0, len(list_in), num_per_part):
            yield list_in[i:i + num_per_part]
    return list(segment_list_to_generator(list_in, num_per_part))


def box_tlwh_to_tlbr(box):
    box = box.copy()
    return [int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])]


def resize_box(box, w_ratio, h_ratio):
    box = box.copy()
    return [int(box[0] * w_ratio), int(box[1] * h_ratio), int(box[2] * w_ratio), int(box[3] * h_ratio)]


def resize_boxes(boxes, w_ratio, h_ratio):
    new_boxes = []
    for box in boxes:
        new_boxes.append(resize_box(box, w_ratio, h_ratio))
    return new_boxes