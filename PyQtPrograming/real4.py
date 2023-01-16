import sys
import os
import time
import shutil
from glob import glob
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, strip_optimizer, xyxy2xywh, LOGGER
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.augmentations import letterbox
### pyinstaller에 묶기 위한 tresh import ###
import seaborn
import yaml
import PIL
import scipy
import utils
import models

##########################################

dir = os.getcwd()
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
FORMATS = IMG_FORMATS + VID_FORMATS


# class LoadImages:
#     # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
#     def __init__(self, path, img_size=640, stride=32, auto=True):
#         p = str(Path(path).resolve())  # os-agnostic absolute path
#         if '*' in p:
#             files = sorted(glob(p, recursive=True))  # glob
#         elif os.path.isdir(p):
#             files = sorted(glob(os.path.join(p, '*.*')))  # dir
#         elif os.path.isfile(p):
#             files = [p]  # files
#         else:
#             raise Exception(f'ERROR: {p} does not exist')
#
#         images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
#         videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
#         ni, nv = len(images), len(videos)
#
#         self.img_size = img_size
#         self.stride = stride
#         self.files = images + videos
#         self.nf = ni + nv  # number of files
#         self.video_flag = [False] * ni + [True] * nv
#         self.mode = 'image'
#         self.auto = auto
#         if any(videos):
#             self.new_video(videos[0])  # new video
#         else:
#             self.cap = None
#         assert self.nf > 0, f'No images or videos found in {p}. ' \
#                             f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'
#
#     def __iter__(self):
#         self.count = 0
#         return self
#
#     def __next__(self):
#         if self.count == self.nf:
#             raise StopIteration
#         path = self.files[self.count]
#
#         if self.video_flag[self.count]:
#             # Read video
#             self.mode = 'video'
#             ret_val, img0 = self.cap.read()
#             if not ret_val:
#                 self.count += 1
#                 self.cap.release()
#                 if self.count == self.nf:  # last video
#                     raise StopIteration
#                 else:
#                     path = self.files[self.count]
#                     self.new_video(path)
#                     ret_val, img0 = self.cap.read()
#
#             self.frame += 1
#             s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
#
#         else:
#             # Read image
#             self.count += 1
#
#             # 변경부분
#             ff = np.fromfile(path, np.uint8) #
#             img0 = cv2.imdecode(ff, cv2.IMREAD_COLOR) #
#             # img0 = cv2.imread(path)  # BGR
#
#             assert img0 is not None, f'Image Not Found {path}'
#             s = f'image {self.count}/{self.nf} {path}: '
#
#         # Padded resize
#         img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
#
#         # Convert
#         img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         img = np.ascontiguousarray(img)
#
#         return path, img, img0, self.cap, s
#
#     def new_video(self, path):
#         self.frame = 0
#         self.cap = cv2.VideoCapture(path)
#         self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     def __len__(self):
#         return self.nf  # number of files

class DetThread(QThread):  # 쓰레드 정의
    send_img = pyqtSignal(np.ndarray)  # 처리 이미지 신호
    send_raw = pyqtSignal(np.ndarray)  # 원본 이미지 신호
    send_statistic = pyqtSignal(dict)  # detecting 결과 신호

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25

    @torch.no_grad()  # detect.py
    def run(self,
            weights=dir + '/yolov5s.pt',  # model.pt path(s)
            source=dir + '/data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=dir + '/runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            unknown=True,
            ):
        source = str(source)

        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        if pt:
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if unknown:
                names.append('Unknown')
            if half:
                model.half()  # to FP16
            if classify:  # second-stage classifier
                modelc = load_classifier(name='resnet50', n=2)  # initialize
                modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
        elif onnx:
            if dnn:
                check_requirements(('opencv-python>=4.5.4',))
                net = cv2.dnn.readNetFromONNX(w)
            else:
                check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
                import onnxruntime
                session = onnxruntime.InferenceSession(w, None)
        else:  # TensorFlow models
            check_requirements(('tensorflow>=2.4.1',))
            import tensorflow as tf
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""),
                                                   [])  # wrapped import
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                model = tf.keras.models.load_model(w)
            elif tflite:
                interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
                int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0
        temp = ""
        # for name in names[:-1]:
        #     exec(f'{name}_list = set([])')
        goranilist = set([])
        wildboarlist = set([])
        humanlist = set([])
        for path, img, im0s, vid_cap, s in dataset:
            if temp is not path:
                # for name in names[:-1]:
                #     exec(f'{name}_cnt = 0')
                cnt_gorani = 0
                cnt_wildboar = 0
                cnt_human = 0
            temp = path

            statistic_dic = {name: 0 for name in names}
            t1 = time_sync()
            if onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            if pt:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(img, augment=augment, visualize=visualize)[0]
            elif onnx:
                if dnn:
                    net.setInput(img)
                    pred = torch.tensor(net.forward())
                else:
                    pred = torch.tensor(
                        session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
            else:  # tensorflow model (tflite, pb, saved_model)
                imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
                if pb:
                    pred = frozen_func(x=tf.constant(imn)).numpy()
                elif saved_model:
                    pred = model(imn, training=False).numpy()
                elif tflite:
                    if int8:
                        scale, zero_point = input_details[0]['quantization']
                        imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                    interpreter.set_tensor(input_details[0]['index'], imn)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    if int8:
                        scale, zero_point = output_details[0]['quantization']
                        pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
                pred[..., 0] *= imgsz[1]  # x
                pred[..., 1] *= imgsz[0]  # y
                pred[..., 2] *= imgsz[1]  # w
                pred[..., 3] *= imgsz[0]  # h
                pred = torch.tensor(pred)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det,
                                       unknown=unknown)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                imr = im0.copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            statistic_dic[names[c]] += 1
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                print(f'{s}Done. ({t3 - t2:.3f}s)')
                # Stream results
                im0 = annotator.result()
                if view_img:
                    p = str(p).split('\\')[-1]

                    # img_array = np.fromfile(str(p), np.uint8)  #한글명 디코딩
                    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    # cv2.imshow(p, im0)
                    # cv2.imshow('raw', imr)
                    # cv2.moveWindow(p, 50, 50)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

                # time.sleep(0.025)
                self.send_img.emit(im0)  # 처리영상
                self.send_raw.emit(imr if isinstance(im0s, np.ndarray) else imr[0])  # 원본영상
                self.send_statistic.emit(statistic_dic)  # detecting 결과
        #
        #     # 파일분리 (하드코딩)
        #     if 'Wildboar' in s:
        #         cnt_wildboar = cnt_wildboar + 1
        #     elif 'Deer' in s:
        #         cnt_gorani = cnt_gorani + 1
        #     elif 'human' in s:
        #         cnt_human = cnt_human + 1
        #
        #     # if not webcam:
        #     ext = path.split('.')[-1].lower()  # 파일 확장자 분리
        #
        #     if ext in FORMATS:  # 사진 리스트 분리
        #         if 'Deer' in s:
        #             goranilist.add(path)
        #         elif 'Wildboar' in s:
        #             wildboarlist.add(path)
        #         elif 'human' in s:
        #             humanlist.add(path)
        #
        #     # if ext in VID_FORMATS:  # 동영상 리스트 분리
        #     #     if cnt_gorani >= 1:
        #     #         goranilist.add(path)
        #     #     elif cnt_wildboar >= 1:
        #     #         wildboarlist.add(path)
        #     #     elif cnt_human >= 1:
        #     #         humanlist.add(path)
        # print(goranilist)
        # print(wildboarlist)
        #
        # if len(wildboarlist) > 0:  # 파일 옮기기
        #     to = './Wildboar'
        #     if not os.path.isdir(to):
        #         os.mkdir(to)
        #     for i in wildboarlist:
        #         try:
        #             shutil.move(i, to)
        #         except:
        #             pass
        #
        # if len(goranilist) > 0:
        #     to = './Deer'
        #     if not os.path.isdir(to):
        #         os.mkdir(to)
        #     for i in goranilist:
        #         try:
        #             shutil.move(i, to)
        #         except:
        #             pass
        #
        # if len(humanlist) > 0:
        #     to = './human'
        #     if not os.path.isdir(to):
        #         os.mkdir(to)
        #     for i in humanlist:
        #         shutil.move(i, to)
        cv2.destroyAllWindows()
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        # if cnt_gorani > 0 and cnt_gorani > cnt_wildboar:
        #     from_ = save_path
        #     to_ = './gorani'
        #     shutil.move(from_, to_)


class MyApp(QMainWindow):  # 메인윈도우 정의

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Detect Yolov5')  # 툴 제목
        self.setWindowIcon(QIcon('logo2.png'))  # 로고 이미지
        self.statusBar()  # 상태바

        # QWidget과 연결
        self.cent_widget = CentWidget()
        self.setCentralWidget(self.cent_widget.tabs)
        self.show()


class CentWidget(QWidget):  # 위젯정의

    def __init__(self):
        super().__init__()
        self.det_thread = DetThread()
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.img2))
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.img1))
        self.det_thread.send_statistic.connect(self.show_statistic)
        griddir = QGridLayout()
        gridopt = QGridLayout()

        font = QFont()
        font.setBold(True)
        font.setPointSize(10)

        # weights : model.pt path(s)
        self.lbl_weight = QLabel(dir + '/last.pt', self)  # 고정값
        self.lbl_weight.setFont(font)
        self.lbl_weight.setStyleSheet('background-color: #FFFFFF')
        self.lbl_weight.setStatusTip('Set model.pt path .pt파일의 경로를 설정합니다.')
        self.lbl_weight.setToolTip('Set model.pt path\n.pt파일의 경로를 설정합니다.')
        btn_weight = QPushButton('Weights', self)
        btn_weight.setFont(font)
        btn_weight.setStatusTip('Set model.pt path .pt파일의 경로를 설정합니다.')
        btn_weight.setToolTip('Set model.pt path\n.pt파일의 경로를 설정합니다.')
        btn_weight.clicked.connect(self.weights)

        # source : file/dir/URL/glob, 0 for webcam
        # self.lbl_source = QLabel(dir, self)
        self.lbl_source = QLabel('D:/project/NationalPark_upgrade/PYQT5/yolov5_master/sample')  # 고정값
        self.lbl_source.setFont(font)
        self.lbl_source.setStyleSheet('background-color: #FFFFFF')
        self.lbl_source.setStatusTip("Set foldername to detect 분류/감지할 '폴더'를 설정합니다. *경로상에 한글이 들어가면 오류가 발생할 수 있음!")
        self.lbl_source.setToolTip("Set foldername to detect\n분류/감지할 '폴더'를 설정합니다.\n*경로상에 한글이 들어가면 오류가 발생할 수 있음!")
        btn_source = QPushButton('Source', self)
        btn_source.setFont(font)
        btn_source.setStatusTip("Set foldername to detect 분류/감지할 '폴더'를 설정합니다. *경로상에 한글이 들어가면 오류가 발생할 수 있음!")
        btn_source.setToolTip("Set foldername to detect\n분류/감지할 '폴더'를 설정합니다.\n*경로상에 한글이 들어가면 오류가 발생할 수 있음!")
        btn_source.clicked.connect(self.source)

        # imgsz : inference size (pixels)
        self.lbl_imgsz = QLabel(self)
        self.lbl_imgsz.setNum(480)
        self.lbl_imgsz.setFont(font)
        self.lbl_imgsz.setStyleSheet('background-color: #FFFFFF')
        self.lbl_imgsz.setStatusTip('Set inference size (pixels) 1~1280')
        self.lbl_imgsz.setToolTip('Set inference size (pixels)\n1~1280')
        btn_imgsz = QPushButton('Image Size', self)
        btn_imgsz.setFont(font)
        # menu = QMenu(self)
        # menu.addAction('160')
        # menu.addAction('320')
        # menu.addAction('480')
        # menu.addAction('640')
        # menu.addAction('960')
        # menu.addAction('1280')
        # menu..connect(self.imgsz)
        # cb = QComboBox(self)
        btn_imgsz.setStatusTip('Set inference size (pixels) 1~1280')
        btn_imgsz.setToolTip('Set inference size (pixels)\n1~1280')
        # btn_imgsz.setMenu(menu)
        # btn_imgsz.menu().connect(self.imgsz)
        # menu.activeAction(self.imgsz)#.connect(self.imgsz)
        btn_imgsz.clicked.connect(self.imgsz)

        # conf_thres : confidence threshold
        self.lbl_conf = QLabel(self)
        self.lbl_conf.setText('70%')
        self.setFont(font)
        self.lbl_conf.setStyleSheet('background-color: #FFFFFF')
        btn_conf = QPushButton('Conf-Thres', self)
        btn_conf.setFont(font)
        btn_conf.setStatusTip('Set confidence(%) threshold 1% ~ 99%')
        btn_conf.setToolTip('Set confidence(%) threshold\n1% ~ 99%')
        btn_conf.clicked.connect(self.conf)

        # iou_thres : NMS IOU threshold
        self.lbl_iou = QLabel(self)
        self.lbl_iou.setText('25%')
        self.setFont(font)
        self.lbl_iou.setStyleSheet('background-color: #FFFFFF')
        btn_iou = QPushButton('Iou-Thres', self)
        btn_iou.setFont(font)
        btn_iou.setStatusTip('NMS IOU(%) threshold 1% ~ 99%')
        btn_iou.setToolTip('NMS IOU(%) threshold\n1% ~ 99%')
        btn_iou.clicked.connect(self.iou)

        # max_det : maximum detections per image
        self.lbl_mxd = QLabel(self)
        self.lbl_mxd.setNum(100)
        self.setFont(font)
        self.lbl_mxd.setStyleSheet('background-color: #FFFFFF')
        btn_mxd = QPushButton('Max-Det', self)
        btn_mxd.setFont(font)
        btn_mxd.setStatusTip('maximum detections per image recommend set under 100')
        btn_mxd.setToolTip('maximum detections per image\nrecommend set under 100')
        btn_mxd.clicked.connect(self.det_num)

        # project : save results to project/name
        self.lbl_prj = QLabel(dir + '\\runs\\detect', self)
        self.lbl_prj.setFont(font)
        self.lbl_prj.setStyleSheet('background-color: #FFFFFF')
        btn_prj = QPushButton('Project', self)
        btn_prj.setFont(font)
        btn_prj.setStatusTip('Save results to Project/Name')
        btn_prj.setToolTip('Save results to Project/Name')
        btn_prj.clicked.connect(self.project)

        # name : save results to project/name
        self.lbl_name = QLabel('exp', self)
        self.lbl_name.setFont(font)
        self.lbl_name.setStyleSheet('background-color: #FFFFFF')
        btn_name = QPushButton('Name', self)
        btn_name.setFont(font)
        btn_name.setStatusTip('Save results to Project/Name')
        btn_name.setToolTip('Save results to Project/Name')
        btn_name.clicked.connect(self.name)

        # line_thickness : bounding box thickness (pixels)
        self.lbl_ltk = QLabel(self)
        self.lbl_ltk.setNum(3)
        self.setFont(font)
        self.lbl_ltk.setStyleSheet('background-color: #FFFFFF')
        btn_ltk = QPushButton('Thickness', self)
        btn_ltk.setFont(font)
        btn_ltk.setStatusTip('Bbox thickness (pixels) Bbox굵기(pixels)를 설정합니다.')
        btn_ltk.setToolTip('Bbox thickness (pixels)\nBbox굵기(pixels)를 설정합니다.')
        btn_ltk.clicked.connect(self.ltk)

        btn_start = QPushButton('Start', self)
        btn_start.setFont(font)
        btn_start.clicked.connect(self.run)

        self.chk_cam = QCheckBox('Webcam')
        self.chk_cam.setFont(font)
        self.chk_cam.setChecked(False)
        self.chk_cam.toggled.connect(self.webcam)

        self.lbl_rst = QLabel()
        # self.lbl_rst.setEnabled(False)
        # font = QFont()
        # font.setFamily("Agency FB")
        # font.setPointSize(11)
        # font.setStyleStrategy(QFont.PreferDefault)
        # self.lbl_rst.setFont(font)
        # self.lbl_rst.setAcceptDrops(False)
        # self.lbl_rst.setAutoFillBackground(False)
        self.lbl_rst.setText('인식결과물')

        self.lbl_dict = QListWidget()
        # font = QFont()
        # font.setPointSize(9)
        # self.lbl_dict.setFont(font)
        # self.lbl_dict.setStyleSheet("background:transparent")
        # self.lbl_dict.setFrameShadow(QFrame.Plain)
        # self.lbl_dict.setProperty("showDropIndicator", True)
        # self.lbl_dict.setObjectName("listWidget")
        self.lbl_raw = QLabel()
        self.lbl_raw.setText('원본이미지')
        self.lbl_prc = QLabel()
        self.lbl_prc.setText('처리이미지')

        self.img1 = QLabel()
        self.img2 = QLabel()

        self.sliconf = QSlider(Qt.Horizontal, self)
        self.sliconf.setRange(1, 99)
        self.sliconf.setSingleStep(1)
        self.sliconf.setValue(45)
        self.sliconf.valueChanged.connect(self.conf_chg)

        self.sliiou = QSlider(Qt.Horizontal, self)
        self.sliiou.setRange(1, 99)
        self.sliiou.setSingleStep(1)
        self.sliiou.setValue(25)
        self.sliiou.valueChanged.connect(self.iou_chg)

        # girddir
        griddir.addWidget(btn_weight, 0, 0, 1, 10)
        griddir.addWidget(btn_source, 1, 0, 1, 10)
        griddir.addWidget(btn_prj, 2, 0, 1, 10)
        griddir.addWidget(btn_name, 3, 0, 1, 10)
        griddir.addWidget(self.lbl_weight, 0, 10, 1, 10)
        griddir.addWidget(self.lbl_source, 1, 10, 1, 10)
        griddir.addWidget(self.chk_cam, 1, 20, 1, 2)
        griddir.addWidget(self.lbl_prj, 2, 10, 1, 10)
        griddir.addWidget(self.lbl_name, 3, 10, 1, 10)
        griddir.addWidget(self.img1, 5, 0, 10, 10)
        griddir.addWidget(self.img2, 5, 10, 10, 10)
        griddir.addWidget(self.lbl_raw, 4, 0, 1, 10)
        griddir.addWidget(self.lbl_prc, 4, 10, 1, 10)
        griddir.addWidget(self.lbl_rst, 4, 20, 1, 2)
        griddir.addWidget(self.lbl_dict, 5, 20, 10, 2)
        griddir.addWidget(btn_start, 15, 8, 1, 6)  # 이거 dir로 옮기면서 이미지 찌그러짐 발생

        # gridopt
        gridopt.addWidget(btn_imgsz, 0, 0, 1, 10)
        gridopt.addWidget(self.lbl_imgsz, 0, 10, 1, 3)
        gridopt.addWidget(btn_conf, 1, 0, 1, 10)
        gridopt.addWidget(self.lbl_conf, 1, 10, 1, 3)
        gridopt.addWidget(self.sliconf, 1, 13, 1, 17)
        gridopt.addWidget(btn_iou, 2, 0, 1, 10)
        gridopt.addWidget(self.lbl_iou, 2, 10, 1, 3)
        gridopt.addWidget(self.sliiou, 2, 13, 1, 17)
        gridopt.addWidget(btn_mxd, 3, 0, 1, 10)
        gridopt.addWidget(self.lbl_mxd, 3, 10, 1, 3)
        gridopt.addWidget(btn_ltk, 4, 0, 1, 10)
        gridopt.addWidget(self.lbl_ltk, 4, 10, 1, 3)
        gridopt.addWidget(self.saveoptions(), 5, 0, 10, 10)
        gridopt.addWidget(self.visualize(), 5, 10, 10, 10)
        gridopt.addWidget(self.category(), 5, 20, 10, 10)

        directory = QWidget()
        directory.setLayout(griddir)

        options = QWidget()
        options.setLayout(gridopt)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(directory, 'Directory')
        self.tabs.addTab(options, 'Options')

    def saveoptions(self):
        groupbox = QGroupBox('Save Options')
        vbox = QVBoxLayout()
        vbox.addWidget(self.save())
        vbox.addWidget(self.savetxt())
        self.chk_savecrop = QCheckBox('Save Crop')
        self.chk_savecrop.setStatusTip('If check, save cropped prediction boxes 체크시 프레임별 예측된 bbox의 사진을 카테고리별로 저장합니다')
        self.chk_savecrop.setToolTip('If check, save cropped prediction boxes\n체크시 프레임별 예측된 bbox의 사진을 카테고리별로 저장합니다')
        vbox.addWidget(self.chk_savecrop)
        groupbox.setLayout(vbox)
        return groupbox

    def save(self):
        self.groupbox1 = QGroupBox('Save')
        self.groupbox1.setCheckable(True)
        self.groupbox1.setChecked(False)
        self.groupbox1.setStatusTip('If check off, do not save images/videos 체크해제시 처리된 이미지나 동영상을 저장하지 않습니다.')
        self.groupbox1.setToolTip('If check off, do not save images/videos\n체크해제시 처리된 이미지나 동영상을 저장하지 않습니다.')
        vbox = QVBoxLayout()
        self.chk_exok = QCheckBox('Exist Ok')
        self.chk_exok.setStatusTip(
            'If check, existing project/name ok, do not increment 체크시 디렉토리를 추가로 생성하지 않습니다. 처리된 영상/이미지의 파일명이 같다면 기존파일을 덮어씁니다.')
        self.chk_exok.setToolTip(
            'If check, existing project/name ok, do not increment\n체크시 디렉토리를 추가로 생성하지 않습니다.\n처리된 영상/이미지의 파일명이 같다면 기존파일을 덮어씁니다.')
        self.groupbox1.toggled.connect(self.signal1)
        vbox.addWidget(self.chk_exok)
        self.groupbox1.setLayout(vbox)
        return self.groupbox1

    def savetxt(self):
        self.groupbox2 = QGroupBox('Save txt')
        self.groupbox2.setCheckable(True)
        self.groupbox2.setChecked(False)
        self.groupbox2.setStatusTip('If check, save results to *.txt 체크해제시 인식 결과(카테고리, bbox좌표)를 txt파일로 저장합니다.')
        self.groupbox2.setToolTip('If check, save results to *.txt\n체크해제시 인식 결과(카테고리, bbox좌표)를 txt파일로 저장합니다.')
        vbox = QVBoxLayout()
        self.chk_saveconf = QCheckBox('Save Conf')
        self.chk_saveconf.setStatusTip('If check, save confidences in --save-txt labels 체크시 신뢰도값을 txt파일에 추가합니다.')
        self.chk_saveconf.setToolTip('If check, save confidences in --save-txt labels\n체크시 신뢰도값을 txt파일에 추가합니다.')
        self.groupbox2.toggled.connect(self.signal2)
        vbox.addWidget(self.chk_saveconf)
        self.groupbox2.setLayout(vbox)
        return self.groupbox2

    # def developermode(self):  # 개발자모드
    #     self.groupbox3 = QGroupBox('Developer Options')
    #     self.groupbox3.setCheckable(True)
    #     self.groupbox3.setChecked(False)
    #     vbox = QVBoxLayout()
    #     chk1 = QCheckBox('Agnostic_nms')
    #     vbox.addWidget(chk1)
    #     chk2 = QCheckBox('Augment')
    #     vbox.addWidget(chk2)
    #     chk3 = QCheckBox('Update')
    #     vbox.addWidget(chk3)
    #     chk4 = QCheckBox('Half')
    #     vbox.addWidget(chk4)
    #     chk5 = QCheckBox('Dnn')
    #     vbox.addWidget(chk5)
    #     chk6 = QCheckBox('Visualize')
    #     vbox.addWidget(chk6)
    #     chk1.setStatusTip('If check, class-agnostic NMS')
    #     chk1.setToolTip('If check, class-agnostic NMS')
    #     chk2.setStatusTip('If check, augmented inference')
    #     chk2.setToolTip('If check, augmented inference')
    #     chk3.setStatusTip('If check, update all models')
    #     chk3.setToolTip('If check, update all models')
    #     chk4.setStatusTip('If check, use FP16 haself.lf-precision inference')
    #     chk4.setToolTip('If check, use FP16 half-precision inference')
    #     chk5.setStatusTip('If check, use OpenCV DNN for ONNX inference')
    #     chk5.setToolTip('If check, use OpenCV DNN for ONNX inference')
    #     chk6.setStatusTip('If check, visualize features')
    #     chk6.setToolTip('If check, visualize features')
    #     self.groupbox3.toggled.connect(self.signal3)
    #     self.groupbox3.setLayout(vbox)
    #     return self.groupbox3

    def category(self):
        gbx = QGroupBox('Category Filter')
        gbx.setStatusTip('If check off, do not classify specific animal 체크 해제시 해당 동물을 분류하지 않습니다.')
        gbx.setToolTip('If check off, do not classify specific animal\n체크 해제시 해당 동물을 분류하지 않습니다.')
        self.chkcat1 = QCheckBox('WildBoar')
        self.chkcat1.setStatusTip('멧돼지')
        self.chkcat1.setToolTip('멧돼지')
        self.chkcat1.setChecked(True)
        self.chkcat2 = QCheckBox('WaterDeer')
        self.chkcat2.setStatusTip('고라니')
        self.chkcat2.setToolTip('고라니')
        self.chkcat2.setChecked(True)
        self.chkcat3 = QCheckBox('HalfMoonBear')
        self.chkcat3.setStatusTip('반달가슴곰')
        self.chkcat3.setToolTip('반달가슴곰')
        self.chkcat3.setChecked(True)
        self.chkcat4 = QCheckBox('Goral')
        self.chkcat4.setStatusTip('산양')
        self.chkcat4.setToolTip('산양')
        self.chkcat4.setChecked(True)
        vbox = QVBoxLayout()
        vbox.addWidget(self.chkcat1)
        vbox.addWidget(self.chkcat2)
        vbox.addWidget(self.chkcat3)
        vbox.addWidget(self.chkcat4)
        gbx.setLayout(vbox)
        return gbx

    def visualize(self):
        gbx = QGroupBox('Visualize Options')
        self.chk_view = QCheckBox('View Image')
        self.chk_view.setChecked(True)
        self.chk_view.setStatusTip('If check, show results 체크시 영상을 직접 화면에 띄웁니다.')
        self.chk_view.setToolTip('If check, show results\n체크시 영상을 직접 화면에 띄웁니다.')
        self.chk_hlbl = QCheckBox('Hide Labels')
        self.chk_hlbl.setStatusTip('If check, hide labels 체크시 처리된 영상이 분류된 종 이름을 띄우지 않습니다.')
        self.chk_hlbl.setToolTip('If check, hide labels\n 체크시 처리된 영상이 분류된 종 이름을 띄우지 않습니다.')
        self.chk_hconf = QCheckBox('Hide Conf')
        self.chk_hconf.setStatusTip('If check, hide confidences 체크시 처리된 영상이 confidence값을 띄우지 않습니다.')
        self.chk_hconf.setToolTip('If check, hide confidences\n체크시 처리된 영상이 confidence값을 띄우지 않습니다.')
        vbox = QVBoxLayout()
        vbox.addWidget(self.chk_view)
        vbox.addWidget(self.chk_hlbl)
        vbox.addWidget(self.chk_hconf)
        gbx.setLayout(vbox)
        return gbx

    def weights(self):
        fname = QFileDialog.getOpenFileName(self, 'Weights', filter='*.pt')
        self.lbl_weight.setText(str(fname[0]))
        self.lbl_weight.adjustSize()

    def source(self):
        fname = QFileDialog.getExistingDirectory(self, 'Source')
        self.lbl_source.setText(str(fname))
        self.lbl_source.adjustSize()

    def webcam(self):
        if self.chk_cam.isChecked():
            self.lbl_source.setNum(0)

    def imgsz(self, num):
        # self.lbl_imgsz.setNum(int(num))
        # self.lbl_imgsz.adjustSize()
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Image Size')
        dlg.setLabelText("inference size (pixels)\n1~1280")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 1280)
        dlg.setIntValue(int(self.lbl_imgsz.text()))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl_imgsz.setNum(num)
            self.lbl_imgsz.adjustSize()

    def conf(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Conf Thres')
        dlg.setLabelText("confidence(%) threshold\n1% ~ 99%")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 99)
        dlg.setIntValue(int(self.lbl_conf.text()[:-1]))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl_conf.setText(str(num) + '%')
            self.lbl_conf.adjustSize()
            self.sliconf.setValue(num)

    def conf_chg(self):
        self.lbl_conf.setText(str(self.sliconf.value()) + '%')

    def iou(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Iou Thres')
        dlg.setLabelText("NMS IOU(%) threshold\n1%~99%")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 99)
        dlg.setIntValue(int(self.lbl_iou.text()[:-1]))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl_iou.setText(str(num) + '%')
            self.lbl_iou.adjustSize()
            self.sliiou.setValue(num)

    def iou_chg(self):
        self.lbl_iou.setText(str(self.sliiou.value()) + '%')

    def det_num(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Max detection')
        dlg.setLabelText("maximum detections per image\n1~9999\nrecommend set under 100")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 9999)
        dlg.setIntValue(int(self.lbl_mxd.text()))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl_mxd.setNum(num)
            self.lbl_mxd.adjustSize()

    def project(self):
        fname = QFileDialog.getExistingDirectory(self, 'Project')
        self.lbl_prj.setText(str(fname))
        self.lbl_prj.adjustSize()

    def name(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.TextInput)
        dlg.setWindowTitle('Name')
        dlg.setLabelText(
            f"save results to Project/Name\n저장할 폴더명\n현재 경로 : {self.lbl_prj.text()}\\{self.lbl_name.text()}")
        dlg.setTextValue(self.lbl_name.text())
        dlg.resize(700, 100)
        ok = dlg.exec_()
        text = dlg.textValue()
        if ok:
            self.lbl_name.setText(str(text))
            self.lbl_name.adjustSize()

    def ltk(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Line Thickness')
        dlg.setLabelText("Bbox Line Thickness (Pixels)\n1~10")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 10)
        dlg.setIntValue(int(self.lbl_ltk.text()))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl_ltk.setNum(num)
            self.lbl_ltk.adjustSize()

    def signal1(self):
        if not self.groupbox1.isChecked():
            self.chk_exok.setCheckState(False)

    def signal2(self):
        if not self.groupbox2.isChecked():
            self.chk_saveconf.setCheckState(False)

    # def signal3(self):  # password
    #     if self.groupbox3.isChecked():
    #         dlg = QInputDialog(self)
    #         dlg.setInputMode(QInputDialog.TextInput)
    #         dlg.setWindowTitle('Enter Password')
    #         dlg.setLabelText("Made by BigleaderTeam")
    #         dlg.resize(700, 100)
    #         ok = dlg.exec_()
    #         text = dlg.textValue()
    #         if ok:
    #             if not str(text) == 'rlacksdlf':
    #                 self.groupbox3.setChecked(False)
    #         else:
    #             self.groupbox3.setChecked(False)

    def run(self):
        li = []
        if self.chkcat1.isChecked():
            li.append(1)
        if self.chkcat2.isChecked():
            li.append(2)
        if self.chkcat3.isChecked():
            li.append(3)
        if self.chkcat4.isChecked():
            li.append(4)
        self.det_thread.run(
            weights=self.lbl_weight.text(),
            source=self.lbl_source.text(),
            imgsz=[int(self.lbl_imgsz.text()), int(self.lbl_imgsz.text())],
            conf_thres=float(int(self.lbl_conf.text()[:-1]) / 100),
            iou_thres=float(int(self.lbl_iou.text()[:-1]) / 100),
            max_det=int(self.lbl_mxd.text()),
            device='',
            view_img=self.chk_view.isChecked(),
            save_txt=self.groupbox2.isChecked(),
            save_conf=self.chk_saveconf.isChecked(),
            save_crop=self.chk_savecrop.isChecked(),
            nosave=not self.groupbox1.isChecked(),
            classes=None,  # if len(li) == 0 else li,
            agnostic_nms=False,
            augment=False,
            visualize=False,
            update=False,
            project=self.lbl_prj.text(),
            name=self.lbl_name.text(),
            exist_ok=self.chk_exok.isChecked(),
            line_thickness=int(self.lbl_ltk.text()),
            hide_labels=self.chk_hlbl.isChecked(),
            hide_conf=self.chk_hconf.isChecked(),
            half=False,
            dnn=False
        )

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.lbl_dict.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.lbl_dict.addItems(results)

        except Exception as e:
            print(repr(e))


# class GroupBox(QGroupBox):  # reverse gbx
#     def paintEvent(self, event):
#         painter = QStylePainter(self)
#         option = QStyleOptionGroupBox()
#         self.initStyleOption(option)
#         if self.isCheckable():
#             option.state &= ~QStyle.State_Off & ~QStyle.State_On
#             option.state |= (
#                 QStyle.State_Off
#                 if self.isChecked()
#                 else QStyle.State_On
#             )
#         painter.drawComplexControl(QStyle.CC_GroupBox, option)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    size = screen.size()
    w = size.width()
    h = size.height()
    ex = MyApp()
    ex.setGeometry(int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h))
    ex.showMaximized()
    sys.exit(app.exec_())
