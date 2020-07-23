import math
import numpy as np
import cv2
from config_utils import CONFIG

from visdom import Visdom
viz = Visdom(server='localhost',port=8097)



def mergeHeatmap(hm, max_width=3):
    width_max = max_width
    hm_class, hm_height, hm_width = hm.shape[:]
    hm_sum = float(hm_class) /width_max
    height_max = math.ceil(hm_sum)
    hm_merge = np.zeros([height_max*hm_height , hm_width * width_max], dtype=np.float32)
    for index in range(len(hm)):
        y_index = int(index / width_max)
        x_index = index % width_max
        hm_merge[y_index * hm_height:(y_index + 1) * hm_height, x_index * hm_width:(x_index + 1) * hm_width] = hm[index].copy()
        hm_merge = cv2.rectangle(hm_merge, (x_index*hm_width, y_index*hm_height), (x_index*hm_width + hm_width, y_index*hm_height + hm_height), (1), 2)
    return hm_merge



def vis(hm, heatmap_gt, images):
    output_detach = hm.detach().cpu().numpy().copy()
    input_detach = heatmap_gt.detach().cpu().numpy().copy()
    image_detach = images.detach().cpu().numpy().copy()

    for index in range(len(output_detach)):
        # if np.max(input_detach[index][18]) >0.8 or np.max(input_detach[index][19]) >0.8 or np.max(input_detach[index][3]) >0.8 or np.max(input_detach[index][4]) >0.8 or np.max(input_detach[index][1]) >0.8 or np.max(input_detach[index][2]) >0.8 :
        hm_pre_merge = mergeHeatmap(output_detach[index])
        hm_gt_merge  = mergeHeatmap(input_detach[index])
    
        hm_pre_merge = hm_pre_merge* 255
        hm_pre_merge = hm_pre_merge.astype(np.uint8)
    
        hm_gt_merge = hm_gt_merge * 255
        hm_gt_merge = hm_gt_merge.astype(np.uint8)
    
        image = image_detach[index]
        image = image.transpose(1, 2, 0)

        image = (image * CONFIG.std + CONFIG.mean) * 255
        image = image.astype(np.uint8)
        image = image[:,:, ::-1]
        image = image.transpose(2,0,1)
    
    
        viz.image(
            image,
            win = 'im',opts = {
                'title':'im',
            }
        )
        viz.image(
            hm_pre_merge,
            win='hm_pre_merge', opts={
                'title': 'hm_pre_merge',
            }
        )
    
        viz.image(
            hm_gt_merge,
            win='hm_gt_merge', opts={
                'title': 'hm_gt_merge',
            }
        )
    
    
        break
        #imgae = cv2.cvtColor(imgae, cv2.COLOR_BGR2RGB)
        # cv2.imshow("hm_gt_merge", hm_gt_merge)
        # cv2.imshow("hm_pre_merge", hm_pre_merge)
        # cv2.imshow("imgae", imgae)
        # cv2.waitKey()
#################################################add show visdom###################################################