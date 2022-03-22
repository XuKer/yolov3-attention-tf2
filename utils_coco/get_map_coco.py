import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils.utils import cvtColor, preprocess_input, resize_image
from yolo import YOLO

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#----------------------------------------------------------------------------#
#   map_mode Used to specify what to calculate when the file is run
#   map_mode 0 represents the entire map calculation process, including obtaining prediction results, calculating map
#   map_mode A value of 1 means that only prediction results are obtained
#   map_mode A value of 2 means that only the calculation map is obtained.
#--------------------------------------------------------------------------#
map_mode            = 0
#-------------------------------------------------------#
#   Points to the validation set label and image path
#-------------------------------------------------------#
cocoGt_path         = 'coco_dataset/annotations/instances_val2017.json'
dataset_img_path    = 'coco_dataset/val2017'
#-------------------------------------------------------#
#   The folder for the result output, the default is map out
#-------------------------------------------------------#
temp_save_path      = 'map_out/coco_eval'

class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   Detect pictures
    #---------------------------------------------------#
    def detect_image(self, image_id, image, results, clsid2catid):
        #---------------------------------------------------------#
        #   Convert the image to an RGB image here to prevent the grayscale image from making errors during prediction.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for identification
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add the batch size dimension and normalize it
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   Feed the image into the network to make predictions!
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape)

        for i, c in enumerate(out_classes):
            result                      = {}
            top, left, bottom, right    = out_boxes[i]

            result["image_id"]      = int(image_id)
            result["category_id"]   = clsid2catid[c]
            result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"]         = float(out_scores[i])
            results.append(result)

        return results

if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    cocoGt      = COCO(cocoGt_path)
    ids         = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    if map_mode == 0 or map_mode == 1:
        yolo = mAP_YOLO(confidence = 0.001, nms_iou = 0.65)

        with open(os.path.join(temp_save_path, 'eval_results.json'),"w") as f:
            results = []
            for image_id in tqdm(ids):
                image_path  = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
                image       = Image.open(image_path)
                results     = yolo.detect_image(image_id, image, results, clsid2catid)
            json.dump(results, f)

    if map_mode == 0 or map_mode == 2:
        cocoDt      = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print("Get map done.")
