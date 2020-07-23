import os
import torch
import cv2
import json
from models import Yolov4
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect




from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def buildCocoGT(dataset):
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco

def formatStats(stats):

    return """
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.3f}
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}
     """.format(*stats)

def get_coco_mAP(gt_dict, pred_list):
    # cocoGt = COCO(gtfile)
    cocoGt = buildCocoGT(gt_dict)
    if pred_list:
        cocoDt = cocoGt.loadRes(pred_list)

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stats = cocoEval.stats
        map_iou0_5 = stats[1]
        info = formatStats(stats)
        return info, map_iou0_5
    else:
        return "result is empty", 0.




def main():
    n_classes = 18
    weightfile = r'D:\DeepBlueProjects\chem-lab\pytorch-YOLOv4-master\checkpoints\Yolov4_epoch6.pth'
    imgfile = r'D:\Data\CMS01_single-end\val\JPEGImages\frontfront_0518.jpg'
    base_dir = r'D:\Data\chem-yolov4\eval-dataset\top'
    gt_path = os.path.join(base_dir, 'gt.json')
    name_id_path = os.path.join(base_dir, 'name_id.json')
    with open(gt_path, 'r') as f:
        gt_dict = json.load(f)
    with open(name_id_path, 'r') as f:
        name_id_dict = json.load(f)

    input_size = (960, 960)

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))


    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。


    model.load_state_dict(pretrained_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    data_txt = os.path.join(base_dir, 'data.txt')
    save_dir = os.path.join(base_dir, 'JPEGImages_pred')
    result_dir = os.path.join(base_dir, 'result_txt')
    with open(data_txt, 'r') as f:
        imgfiles = f.readlines()

    box_list = []
    for imgfile in imgfiles:

        img = cv2.imread(imgfile.strip('\n'))
        img_h, img_w, _ = img.shape

        img_name = imgfile.split('\\')[-1].strip('\n')
        img_id = name_id_dict[img_name]
        result_txt = os.path.join(result_dir, img_name[:-4] + '.txt')
        result_f = open(result_txt, 'w')
        # Inference input size is 416*416 does not mean training size is the same
        # Training size could be 608*608 or even other sizes
        # Optional inference sizes:
        #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
        #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
        sized = cv2.resize(img, (input_size[1], input_size[0]))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)


        # for i in range(2):  # This 'for' loop is for speed check
                            # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.01, 0.3, use_cuda)

        for box in boxes[0]:
            x1 = int((box[0] - box[2] / 2.0) * img_w)
            y1 = int((box[1] - box[3] / 2.0) * img_h)
            x2 = int((box[0] + box[2] / 2.0) * img_w)
            y2 = int((box[1] + box[3] / 2.0) * img_h)
            w = x2 - x1
            h = y2 - y1

            if len(box) >= 7:
                cls_conf = box[5]
                cls_id = box[6]
                box_list.append(
                    {"image_id": img_id, "category_id": int(cls_id), "bbox": [x1, y1, w, h],
                     "score": float(cls_conf)})
                string = ','.join([str(cls_id), str(x1), str(y1), str(x2), str(y2), str(cls_conf)]) + '\n'
                result_f.write(string)

                if cls_conf > 0.3:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                                 (255, 0, 255), 1)
                    cv2.putText(img, str(cls_id),
                                (int(x1 + 10), int(y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
                    cv2.putText(img, str(round(cls_conf, 3)), (int(x1 + 30), int(y1 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 240, 0), 1)
            else:
                print('????')


        result_f.close()
        namesfile = 'data/chem.names'
        class_names = load_class_names(namesfile)
        save_name = os.path.join(save_dir, img_name)
        plot_boxes_cv2(img, boxes[0], save_name, class_names)
        # cv2.imshow('result', img)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    info, map_iou0_5 = get_coco_mAP(gt_dict, box_list)
    # print("---base_eval---epoch%d"%real_epoch)
    print(info)


if __name__ == '__main__':
    main()