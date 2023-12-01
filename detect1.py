import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

def number_recognition(crop_path,model1,stride1, names1, pt1):
    imgsz=(640, 640)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device=''
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_csv=False  # save results in CSV format
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project=ROOT / 'runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    vid_stride=1
    
    global number_plate_string
    imgsz = check_img_size(imgsz, s=stride1)  # check image size
 
    # Dataloader
    bs1 = 1  # batch_size
    dataset1 = LoadImages(crop_path, img_size=imgsz, stride=stride1, auto=pt1, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs1, [None] * bs1

    # Run inference
    model1.warmup(imgsz=(1 if pt1 or model1.triton else bs1, 3, *imgsz))  # warmup
    seen1, windows1, dt1 = 0, [], (Profile(), Profile(), Profile())

    for path1, im1, im0s1, vid_cap1, s1 in dataset1:
        
        with dt1[0]:
            im1 = torch.from_numpy(im1).to(model1.device)
            im1 = im1.half() if model1.fp16 else im1.float()  # uint8 to fp16/32
            im1 /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im1.shape) == 3:
                im1 = im1[None]  # expand for batch dim

        # Inference
        with dt1[1]:
            visualize = increment_path(save_dir / Path(path1).stem, mkdir=True) if visualize else False
            pred1 = model1(im1, augment=augment, visualize=visualize)

        # NMS
        with dt1[2]:
            pred1 = non_max_suppression(pred1, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        
        # Process predictions
        for i1, det1 in enumerate(pred1):  # per image
            seen1 += 1
            p1, im01, frame1 = path1, im0s1.copy(), getattr(dataset1, 'frame', 0)

            gn1 = torch.tensor(im01.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc1 = im01.copy() if save_crop else im01  # for save_crop
            annotator1 = Annotator(im01, line_width=line_thickness, example=str(names1))

            if len(det1):
                
                # Rescale boxes from img_size to im0 size
                det1[:, :4] = scale_boxes(im1.shape[2:], det1[:, :4], im01.shape).round()
                number = ()
                my_dict = {}
                # cv2.imshow("crop",im01)

                # Print results
                for c1 in det1[:, 5].unique():
                    n1 = (det1[:, 5] == c1).sum()  # detections per class
                    s1 += f"{n1} {names1[int(c1)]}{'s1' * (n1 > 1)}, "  # add to string


                # Write results
                for *xyxy, conf, cls in reversed(det1):
                    c = int(cls)  # integer class
                    label1 = names1[c] if hide_conf else f'{names1[c]}'
                    confidence1 = float(conf)
                    confidence_str1 = f'{confidence1:.2f}'
                    x_min,y_min,x_max,y_max = xyxy
                    my_dict[int(x_min)] = str(label1)
                    number += (int(x_min),) 

                number = tuple(sorted(number))   
                number_plate = []
                for x_min in number:
                    for key,value in my_dict.items():
                        if x_min == key:
                            number_plate +=str(value)
                number_plate_string = ''.join(number_plate)     
                # print(" license plate number: ",number_plate_string) 
                return number_plate_string
               

# if __name__ == '__main__':
#     crop_path = Path.cwd()/'runs/detect/exp13/crops/license_plate'
#     print("cureent path: ",crop_path)
#     half = True
#     dnn = True
#     device=''
#     data= Path.cwd() / 'data/coco128.yaml'
#     weights = Path.cwd()/'best.pt'
#     device = select_device(device)
#     model1 = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride1, names1, pt1 = model1.stride, model1.names, model1.pt
#     print("Model2 loaded")

#     number = number_recognition(crop_path,model1,stride1, names1, pt1)
#     print(number)
