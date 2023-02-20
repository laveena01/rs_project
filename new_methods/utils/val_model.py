import torch
import os
import sys
import cv2
from CAM_tools import *
from utils.functions import *
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from expr.train import device

# def write_result(model, img_path, image, result_path):
#     image = image.to(device)
#     confidence = model(image)
#     cam = model.get_cam_maps()
#     # cam = model.get_child_maps()
#     # cam = model.get_parent_maps()
#     cam = cam.squeeze(0).cpu().detach().numpy()
#     res = np.zeros(cam.shape).sum(0)
#     for idx in range(len(class_names)):
#         if confidence.data[0, idx] > 1.00:
#             print('[class_idx: %d] %s (%.2f)' % (idx, class_names[idx], confidence[0, idx]), end=' ')
#             res = res + cam[idx]
#     print()
#     cam = res
#     cam = cam - np.min(cam)
#     cam_img = cam / np.max(cam)
#     cam_img = np.uint8(255 * cam_img)
#     img = cv2.imread(img_path)
#     height, width, _ = img.shape
#     heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)
#     result = heatmap * 0.3 + img * 0.5
#     cv2.imwrite(result_path, result)

def generate_detectresult(model_ft, img_name, inputs, img_path,
                          dst_detections, categories):
    bbox_res4 = []
    confidence = model_ft(inputs)
    _, cam = model_ft.get_salience_maps()
    cam = cam.cpu().data.numpy()
    # generate layer3 map
    # the result is compare with layer4
    # final result is combine with layer3 and layer4


    # generate layer4 map
    for idx in range(len(categories)):
        if confidence.data[0, idx] > 0.00:
            temp = generate_box(idx, cam, categories, img_path, confidence.data[0, idx])
            for t in temp:
                bbox_res4.append(t)

    # soted the result by confidence
    bbox_res4 = sorted(bbox_res4, key=lambda bndresult: bndresult.confidence, reverse=True)

    # generate bbox layer3
    cam = generateCAM(img_path, _)
    bbox_res3 = generateBBox(cam)

    # marege layer3 and layer4 result
    res = mergeCAM(bbox_res3, bbox_res4)


    # visulise result
    # im = cv2.imread(img_path)
    # for t in  bbox_res3:
    #     cv2.rectangle(im, (t.x1, t.y1), (t.x2, t.y2), (0, 255, 255), 2)
    # for t in bbox_res4:
    #     cv2.rectangle(im, (t.x1, t.y1), (t.x2, t.y2), (0, 255, 0), 3)
    #
    #     cv2.putText(im, t.categories + ' ' + str(t.confidence), (t.x1+15, t.y1+15), 1, 1, (0, 0, 255), 1)
    # cv2.namedWindow("img")
    # cv2.moveWindow("img", 0, 0)
    #
    # cv2.imshow("img", im)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()

    # write result to file
    dst_path = os.path.join(dst_detections, img_name+'.txt')
    with open(dst_path, "w") as f:
        for r in bbox_res4:
            f.writelines(str(r))
        f.close()

    dst_path1 = os.path.join(dst_detections+'1', img_name + '.txt')
    with open(dst_path1, "w") as f1:
        for r in res:
            f1.writelines(str(r))
        f1.close()


def write_result_txt(file_path, dst_path, categories):
    res = []
    with open(file_path, 'r') as f:
        contents = f.read()
        objects = contents.split('\n')
        for i in range(objects.count('')):
            objects.remove('')
        for objecto in objects:
            xmin = objecto.split(',')[0]
            xmin = xmin.split('(')[1]
            xmin = int(xmin.strip())

            ymin = objecto.split(',')[1]
            ymin = ymin.split(')')[0]
            ymin = int(ymin.strip())

            xmax = objecto.split(',')[2]
            xmax = xmax.split('(')[1]
            xmax = int(xmax.strip())

            ymax = objecto.split(',')[3]
            ymax = ymax.split(')')[0]
            ymax = int(ymax.strip())
            name = objecto.split(',')[4]
            name = name.strip()
            if int(name) == 10:
                break
            name = categories[int(name)-1]
            res.append(name + ' ' + str(xmin) + ' ' + str(ymin) + ' '
                       + str(xmax) + ' ' + str(ymax)+ '\n')
    print(res)
    with open(dst_path, "w") as f:
        for r in res:
            f.writelines(r)

def write_result_xml(file_path, dst_path, categories):
    node = ET.parse(file_path).getroot()
    res = []
    target = parse_voc_xml(node)
    objects = target['annotation']['object']
    if not isinstance(objects, list):
        name = objects['name']
        xmin = objects['bndbox']['xmin']
        xmax = objects['bndbox']['xmax']
        ymin = objects['bndbox']['ymin']
        ymax = objects['bndbox']['ymax']
        res.append(name + ' ' + str(xmin) + ' ' + str(ymin) + ' '
                   + str(xmax) + ' ' + str(ymax)+ '\n')

    else:
        for object in objects:
            name = object['name']
            xmin = object['bndbox']['xmin']
            xmax = object['bndbox']['xmax']
            ymin = object['bndbox']['ymin']
            ymax = object['bndbox']['ymax']
            res.append(name + ' ' + str(xmin) + ' ' + str(ymin) + ' '
                       + str(xmax) + ' ' + str(ymax) + '\n')

    print(res)
    with open(dst_path, "w") as f:
        for r in res:
            f.writelines(r)

def generate_groundtruth(image, path, name='DIOR', categories=None):
    if name == 'VHR10':
        groundtruth_path = r'D:\PycharmProjects\dataset\NWPU VHR-10 dataset_split_1\ground truth'
        write_result_txt(os.path.join(groundtruth_path, image+'.txt'),
                         os.path.join(path, image+'.txt'), categories)
    elif name == 'DIOR':
        groundtruth_path = r'D:\PycharmProjects\dataset\DIOR\Annotations'
        write_result_xml(os.path.join(groundtruth_path, image+'.xml'),
                         os.path.join(path, image+'.txt'), categories)

def del_file(path):
    del_list = os.listdir(path)
    for f in del_list:
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    name = 'DIOR'
    select_model = 'DA_PAM'
    k = 8
    cos = 0.02

    # dst_groundtruths = r'D:\PycharmProjects\Object-Detection-Metrics\groundtruths'
    dst_detections = '/data/xxxujian/PycharmProjects/Object-Detection-Metrics/detections_' + name

    from utils.load_data import get_dataloader
    if name == 'DIOR':
        root = '/data/xxxujian/PycharmProjects/dataset/DIOR/JPEGImages-test'
        path = 'model_trained/params_multi20_'
    else:
        root = '/data/xxxujian/PycharmProjects/dataset/NWPU VHR-10 dataset_split_1/test'
        path = 'model_trained/params_multi10_'

    dataloaders, dataset_sizes, categories = get_dataloader(name=name, split='test',
                                                            shuffle=False, batch_size=1)

    if select_model == 'CAM':
        from model.resnet import model
        model_ft = model(pretrained=True, num_classes=len(categories))
        model_ft = model_ft.to(device)
        model_path = path + select_model + '0_new.pkl'

        model_ft.load_state_dict(torch.load(model_path, map_location={'cuda:3': 'cuda:4'}))
    elif select_model == 'DA':
        from model.resnet_DA import model
        model_ft = model(pretrained=True, num_classes=len(categories), cos_alpha=cos, num_maps=k)
        model_path = path + select_model + '_' + str(cos) + '_' + str(k) + '_0_' + 'pam.pkl'
        model_ft = model_ft.to(device)
        model_ft.load_state_dict(torch.load(model_path, map_location={'cuda:3': 'cuda:3'}))

    elif select_model == 'ACoL':
        from model.resnet_ACoL import model
        model_ft = model(pretrained=True, num_classes=len(categories))
        model_ft = model_ft.to(device)
        model_path = path + select_model + '0_new.pkl'
        model_path = path + select_model + '_' + str(cos) + '_' + str(k) + '_1_' + 'all.pkl'
        model_ft.load_state_dict(torch.load(model_path, map_location={'cuda:3': 'cuda:3'}))
    else:
        from model.resnet_DA_PAM import model
        model_path = path + select_model + '_' + str(cos) + '_' + str(k) + '_0_' + 'new.pkl'
        model_ft = model(pretrained=True, num_classes=len(categories), cos_alpha=cos, num_maps=k)
        model_ft = model_ft.to(device)
        model_ft.load_state_dict(torch.load(model_path, map_location={'cuda:3': 'cuda:4'}))

    model_ft.eval()

    print(categories)
    del_file(dst_detections)
    for batch_idx, (img_path, inputs, targets) in enumerate(dataloaders):
        inputs, targets = inputs.to(device), targets.to(device)
        print(img_path[0].split('/')[-1].split('.')[0])

        img_name = img_path[0].split('/')[-1].split('.')[0]
        img_path = os.path.join(root, img_name+'.jpg')
        # generate groundtruth
        # generate_groundtruth(img_name,
        #                 dst_groundtruths, name=name, categories=categories)
        #
        # generate detect result
        generate_detectresult(model_ft, img_name, inputs, img_path,
                              dst_detections, categories)



