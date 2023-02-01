import collections
import numpy as np
import cv2
from utils.bndresult import bndresult
def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict

def norm_atten_map(attention_map):
    min_val = np.min(attention_map)
    max_val = np.max(attention_map)
    atten_norm = (attention_map - min_val) / (max_val - min_val)
    return atten_norm


def generate_box(idx, root_map, categories, img_path, confidence):

    res = []

    img = cv2.imread(img_path)


    # threshold = 0.2
    h, w, _ = np.shape(img)
    root_map_ = root_map[0, idx, :, :]

    cam_map_ = norm_atten_map(root_map_)
    cam_img = np.uint8(255 * cam_map_)
    cam_map_cls = cv2.resize(cam_img, dsize=(w, h))

    fg_map, thresh = cv2.threshold(cam_map_cls.copy(), 0, 100,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)


    # fg_map = cam_map_cls >= threshold
    #
    # binary = np.zeros((h, w), dtype=np.uint8)
    # binary.fill(255)
    # binary = binary * fg_map
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        temp = bndresult(x, y, x+w, y+h, categories[idx], confidence.item())
        res.append(temp)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # cv2.putText(img, categories[idx], (x+15, y+15), 1, 1, (255, 255, 0))

    # cv2.namedWindow("img")
    # cv2.moveWindow("img",0, 0)
    #
    # cv2.imshow("img", img)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    return res







