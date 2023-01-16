import sys
import os

import cv2

from yolov5_master import detect
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
### pyinstaller에 묶기 위한 tresh import ###
from PyQt5.QtCore import *
import seaborn
import yaml
import PIL
import scipy
import utils
import models

##########################################

dir = os.getcwd()

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Detect YoloV5')  # 툴 제목
        self.setWindowIcon(QIcon('logo2.png'))  # 로고 이미지
        self.statusBar()

        # QWidget과 연결
        self.cent_widget = CentWidget()
        self.setCentralWidget(self.cent_widget.tabs)
        self.show()


class CentWidget(QWidget):

    def __init__(self):
        super().__init__()

        griddir = QGridLayout()
        gridopt = QGridLayout()
        # self.setLayout(grid)

        font = QFont()
        font.setBold(True)
        font.setPointSize(10)

        # weights : model.pt path(s)
        self.lbl1 = QLabel(dir + '\model.pt', self)
        self.lbl1.setFont(font)
        self.lbl1.setStyleSheet('background-color: #FFFFFF')
        self.lbl1.setStatusTip('Set model.pt path .pt파일의 경로를 설정합니다.')
        self.lbl1.setToolTip('Set model.pt path\n.pt파일의 경로를 설정합니다.')
        btn1 = QPushButton('Weights', self)
        btn1.setFont(font)
        btn1.setStatusTip('Set model.pt path .pt파일의 경로를 설정합니다.')
        btn1.setToolTip('Set model.pt path\n.pt파일의 경로를 설정합니다.')
        btn1.clicked.connect(self.weights)

        # source : file/dir/URL/glob, 0 for webcam
        self.lbl2 = QLabel(dir, self)
        self.lbl2.setFont(font)
        self.lbl2.setStyleSheet('background-color: #FFFFFF')
        self.lbl2.setStatusTip("Set foldername to detect 분류/감지할 '폴더'를 설정합니다. *경로상에 한글이 들어가면 오류가 발생할 수 있음!")
        self.lbl2.setToolTip("Set foldername to detect\n분류/감지할 '폴더'를 설정합니다.\n*경로상에 한글이 들어가면 오류가 발생할 수 있음!")
        btn2 = QPushButton('Source', self)
        btn2.setFont(font)
        btn2.setStatusTip("Set foldername to detect 분류/감지할 '폴더'를 설정합니다. *경로상에 한글이 들어가면 오류가 발생할 수 있음!")
        btn2.setToolTip("Set foldername to detect\n분류/감지할 '폴더'를 설정합니다.\n*경로상에 한글이 들어가면 오류가 발생할 수 있음!")
        btn2.clicked.connect(self.source)

        # imgsz : inference size (pixels)
        self.lbl3 = QLabel(self)
        self.lbl3.setNum(480)
        self.lbl3.setFont(font)
        self.lbl3.setStyleSheet('background-color: #FFFFFF')
        self.lbl3.setStatusTip('Set inference size (pixels) 1~1280')
        self.lbl3.setToolTip('Set inference size (pixels)\n1~1280')
        btn3 = QPushButton('Image Size', self)
        btn3.setFont(font)
        # menu = QMenu(self)
        # menu.addAction('160')
        # menu.addAction('320')
        # menu.addAction('480')
        # menu.addAction('640')
        # menu.addAction('960')
        # menu.addAction('1280')
        # menu..connect(self.imgsz)
        # cb = QComboBox(self)
        btn3.setStatusTip('Set inference size (pixels) 1~1280')
        btn3.setToolTip('Set inference size (pixels)\n1~1280')
        # btn3.setMenu(menu)
        # btn3.menu().connect(self.imgsz)
        # menu.activeAction(self.imgsz)#.connect(self.imgsz)
        btn3.clicked.connect(self.imgsz)

        # conf_thres : confidence threshold
        self.lbl4 = QLabel(self)
        self.lbl4.setText('45%')
        self.setFont(font)
        self.lbl4.setStyleSheet('background-color: #FFFFFF')
        btn4 = QPushButton('Conf-Thres', self)
        btn4.setFont(font)
        btn4.setStatusTip('Set confidence(%) threshold 1% ~ 99%')
        btn4.setToolTip('Set confidence(%) threshold\n1% ~ 99%')
        btn4.clicked.connect(self.conf)

        # iou_thres : NMS IOU threshold
        self.lbl5 = QLabel(self)
        self.lbl5.setText('25%')
        self.setFont(font)
        self.lbl5.setStyleSheet('background-color: #FFFFFF')
        btn5 = QPushButton('Iou-Thres', self)
        btn5.setFont(font)
        btn5.setStatusTip('NMS IOU(%) threshold 1% ~ 99%')
        btn5.setToolTip('NMS IOU(%) threshold\n1% ~ 99%')
        btn5.clicked.connect(self.iou)

        # max_det : maximum detections per image
        self.lbl6 = QLabel(self)
        self.lbl6.setNum(100)
        self.setFont(font)
        self.lbl6.setStyleSheet('background-color: #FFFFFF')
        btn6 = QPushButton('Max-Det', self)
        btn6.setFont(font)
        btn6.setStatusTip('maximum detections per image recommend set under 100')
        btn6.setToolTip('maximum detections per image\nrecommend set under 100')
        btn6.clicked.connect(self.det_num)

        # project : save results to project/name
        self.lbl7 = QLabel(dir + '\\runs\\detect', self)
        self.lbl7.setFont(font)
        self.lbl7.setStyleSheet('background-color: #FFFFFF')
        btn7 = QPushButton('Project', self)
        btn7.setFont(font)
        btn7.setStatusTip('Save results to Project/Name')
        btn7.setToolTip('Save results to Project/Name')
        btn7.clicked.connect(self.project)

        # name : save results to project/name
        self.lbl8 = QLabel('exp', self)
        self.lbl8.setFont(font)
        self.lbl8.setStyleSheet('background-color: #FFFFFF')
        btn8 = QPushButton('Name', self)
        btn8.setFont(font)
        btn8.setStatusTip('Save results to Project/Name')
        btn8.setToolTip('Save results to Project/Name')
        btn8.clicked.connect(self.name)

        # line_thickness : bounding box thickness (pixels)
        self.lbl9 = QLabel(self)
        self.lbl9.setNum(3)
        self.setFont(font)
        self.lbl9.setStyleSheet('background-color: #FFFFFF')
        btn9 = QPushButton('Thickness', self)
        btn9.setFont(font)
        btn9.setStatusTip('Bbox thickness (pixels) Bbox굵기(pixels)를 설정합니다.')
        btn9.setToolTip('Bbox thickness (pixels)\nBbox굵기(pixels)를 설정합니다.')
        btn9.clicked.connect(self.ltk)

        btn10 = QPushButton('Start', self)
        btn10.setFont(font)
        btn10.clicked.connect(self.run)

        self.chk99 = QCheckBox('Webcam')
        self.chk99.setFont(font)
        self.chk99.setChecked(False)
        self.chk99.toggled.connect(self.webcam)


        pixmap = QPixmap('background.jpg')
        self.img1 = QLabel()
        self.img1.setPixmap(pixmap)
        self.img2 = QLabel()
        self.img2.setPixmap(pixmap)

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

        griddir.addWidget(self.img1, 20, 0)
        griddir.addWidget(self.img2, 20, 1)
        gridopt.addWidget(self.sliconf, 3, 2)
        gridopt.addWidget(self.sliiou, 4, 2)

        # Gridbox
        griddir.addWidget(btn1, 0, 0)
        griddir.addWidget(btn2, 1, 0)
        gridopt.addWidget(btn3, 2, 0)
        gridopt.addWidget(btn4, 3, 0)
        gridopt.addWidget(btn5, 4, 0)
        gridopt.addWidget(btn6, 5, 0)
        griddir.addWidget(btn7, 6, 0)
        griddir.addWidget(btn8, 7, 0)
        gridopt.addWidget(btn9, 8, 0)
        griddir.addWidget(self.lbl1, 0, 1)
        griddir.addWidget(self.lbl2, 1, 1)
        griddir.addWidget(self.chk99, 1, 2)
        gridopt.addWidget(self.lbl3, 2, 1)
        gridopt.addWidget(self.lbl4, 3, 1)
        gridopt.addWidget(self.lbl5, 4, 1)
        gridopt.addWidget(self.lbl6, 5, 1)
        griddir.addWidget(self.lbl7, 6, 1)
        griddir.addWidget(self.lbl8, 7, 1)
        gridopt.addWidget(self.lbl9, 8, 1)
        gridopt.addWidget(self.SaveOptions(), 9, 0)
        gridopt.addWidget(self.category(), 9, 2)
        gridopt.addWidget(self.visualize(), 9, 1)
        gridopt.addWidget(btn10, 10, 1)

        directory = QWidget()
        directory.setLayout(griddir)

        options = QWidget()
        options.setLayout(gridopt)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(directory, 'Directory')
        self.tabs.addTab(options, 'Options')

    def SaveOptions(self):
        groupbox = QGroupBox('Save Options')
        vbox = QVBoxLayout()
        vbox.addWidget(self.save())
        vbox.addWidget(self.savetxt())
        self.chk3 = QCheckBox('Save Crop')
        self.chk3.setStatusTip('If check, save cropped prediction boxes 체크시 프레임별 예측된 bbox의 사진을 카테고리별로 저장합니다')
        self.chk3.setToolTip('If check, save cropped prediction boxes\n체크시 프레임별 예측된 bbox의 사진을 카테고리별로 저장합니다')
        vbox.addWidget(self.chk3)
        groupbox.setLayout(vbox)
        return groupbox

    def save(self):
        self.groupbox1 = QGroupBox('Save')
        self.groupbox1.setCheckable(True)
        self.groupbox1.setChecked(True)
        self.groupbox1.setStatusTip('If check off, do not save images/videos 체크해제시 처리된 이미지나 동영상을 저장하지 않습니다.')
        self.groupbox1.setToolTip('If check off, do not save images/videos\n체크해제시 처리된 이미지나 동영상을 저장하지 않습니다.')
        vbox = QVBoxLayout()
        self.chk1 = QCheckBox('Exist Ok')
        self.chk1.setStatusTip(
            'If check, existing project/name ok, do not increment 체크시 디렉토리를 추가로 생성하지 않습니다. 처리된 영상/이미지의 파일명이 같다면 기존파일을 덮어씁니다.')
        self.chk1.setToolTip(
            'If check, existing project/name ok, do not increment\n체크시 디렉토리를 추가로 생성하지 않습니다.\n처리된 영상/이미지의 파일명이 같다면 기존파일을 덮어씁니다.')
        self.groupbox1.toggled.connect(self.signal1)
        vbox.addWidget(self.chk1)
        self.groupbox1.setLayout(vbox)
        return self.groupbox1

    def savetxt(self):
        self.groupbox2 = QGroupBox('Save txt')
        self.groupbox2.setCheckable(True)
        self.groupbox2.setChecked(False)
        self.groupbox2.setStatusTip('If check, save results to *.txt 체크해제시 인식 결과(카테고리, bbox좌표)를 txt파일로 저장합니다.')
        self.groupbox2.setToolTip('If check, save results to *.txt\n체크해제시 인식 결과(카테고리, bbox좌표)를 txt파일로 저장합니다.')
        vbox = QVBoxLayout()
        self.chk2 = QCheckBox('Save Conf')
        self.chk2.setStatusTip('If check, save confidences in --save-txt labels 체크시 신뢰도값을 txt파일에 추가합니다.')
        self.chk2.setToolTip('If check, save confidences in --save-txt labels\n체크시 신뢰도값을 txt파일에 추가합니다.')
        self.groupbox2.toggled.connect(self.signal2)
        vbox.addWidget(self.chk2)
        self.groupbox2.setLayout(vbox)
        return self.groupbox2

    def Developermode(self):
        self.groupbox3 = QGroupBox('Developer Options')
        self.groupbox3.setCheckable(True)
        self.groupbox3.setChecked(False)
        vbox = QVBoxLayout()
        chk1 = QCheckBox('Agnostic_nms')
        vbox.addWidget(chk1)
        chk2 = QCheckBox('Augment')
        vbox.addWidget(chk2)
        chk3 = QCheckBox('Update')
        vbox.addWidget(chk3)
        chk4 = QCheckBox('Half')
        vbox.addWidget(chk4)
        chk5 = QCheckBox('Dnn')
        vbox.addWidget(chk5)
        chk6 = QCheckBox('Visualize')
        vbox.addWidget(chk6)
        chk1.setStatusTip('If check, class-agnostic NMS')
        chk1.setToolTip('If check, class-agnostic NMS')
        chk2.setStatusTip('If check, augmented inference')
        chk2.setToolTip('If check, augmented inference')
        chk3.setStatusTip('If check, update all models')
        chk3.setToolTip('If check, update all models')
        chk4.setStatusTip('If check, use FP16 haself.lf-precision inference')
        chk4.setToolTip('If check, use FP16 half-precision inference')
        chk5.setStatusTip('If check, use OpenCV DNN for ONNX inference')
        chk5.setToolTip('If check, use OpenCV DNN for ONNX inference')
        chk6.setStatusTip('If check, visualize features')
        chk6.setToolTip('If check, visualize features')
        self.groupbox3.toggled.connect(self.signal3)
        self.groupbox3.setLayout(vbox)
        return self.groupbox3

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
        self.chk4 = QCheckBox('View Image')
        self.chk4.setStatusTip('If check, show results 체크시 영상을 직접 화면에 띄웁니다.')
        self.chk4.setToolTip('If check, show results\n체크시 영상을 직접 화면에 띄웁니다.')
        self.chk5 = QCheckBox('Hide Labels')
        self.chk5.setStatusTip('If check, hide labels 체크시 처리된 영상이 분류된 종 이름을 띄우지 않습니다.')
        self.chk5.setToolTip('If check, hide labels\n 체크시 처리된 영상이 분류된 종 이름을 띄우지 않습니다.')
        self.chk6 = QCheckBox('Hide Conf')
        self.chk6.setStatusTip('If check, hide confidences 체크시 처리된 영상이 confidence값을 띄우지 않습니다.')
        self.chk6.setToolTip('If check, hide confidences\n체크시 처리된 영상이 confidence값을 띄우지 않습니다.')
        vbox = QVBoxLayout()
        vbox.addWidget(self.chk4)
        vbox.addWidget(self.chk5)
        vbox.addWidget(self.chk6)
        gbx.setLayout(vbox)
        return gbx

    def weights(self):
        fname = QFileDialog.getOpenFileName(self, 'Weights', filter='*.pt')
        self.lbl1.setText(str(fname[0]))
        self.lbl1.adjustSize()

    def source(self):
        fname = QFileDialog.getExistingDirectory(self, 'Source')
        self.lbl2.setText(str(fname))
        self.lbl2.adjustSize()

    def webcam(self):
        if self.chk99.isChecked():
            self.lbl2.setNum(0)

    def imgsz(self, num):
        # self.lbl3.setNum(int(num))
        # self.lbl3.adjustSize()
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Image Size')
        dlg.setLabelText("inference size (pixels)\n1~1280")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 1280)
        dlg.setIntValue(int(self.lbl3.text()))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl3.setNum(num)
            self.lbl3.adjustSize()

    def conf(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Conf Thres')
        dlg.setLabelText("confidence(%) threshold\n1% ~ 99%")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 99)
        dlg.setIntValue(int(self.lbl4.text()[:-1]))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl4.setText(str(num)+'%')
            self.lbl4.adjustSize()
            self.sliconf.setValue(num)

    def conf_chg(self):
        self.lbl4.setText(str(self.sliconf.value())+'%')

    def iou(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Iou Thres')
        dlg.setLabelText("NMS IOU(%) threshold\n1%~99%")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 99)
        dlg.setIntValue(int(self.lbl5.text()[:-1]))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl5.setText(str(num)+'%')
            self.lbl5.adjustSize()
            self.sliiou.setValue(num)

    def iou_chg(self):
        self.lbl5.setText(str(self.sliiou.value())+'%')

    def det_num(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Max detection')
        dlg.setLabelText("maximum detections per image\n1~9999\nrecommend set under 100")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 9999)
        dlg.setIntValue(int(self.lbl6.text()))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl6.setNum(num)
            self.lbl6.adjustSize()

    def project(self):
        fname = QFileDialog.getExistingDirectory(self, 'Project')
        self.lbl7.setText(str(fname))
        self.lbl7.adjustSize()

    def name(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.TextInput)
        dlg.setWindowTitle('Name')
        dlg.setLabelText(f"save results to Project/Name\n저장할 폴더명\n현재 경로 : {self.lbl7.text()}\\{self.lbl8.text()}")
        dlg.setTextValue(self.lbl8.text())
        dlg.resize(700, 100)
        ok = dlg.exec_()
        text = dlg.textValue()
        if ok:
            self.lbl8.setText(str(text))
            self.lbl8.adjustSize()

    def ltk(self):
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setWindowTitle('Line Thickness')
        dlg.setLabelText("Bbox Line Thickness (Pixels)\n1~10")
        dlg.resize(500, 100)
        dlg.setIntRange(1, 10)
        dlg.setIntValue(int(self.lbl9.text()))
        ok = dlg.exec_()
        num = dlg.intValue()
        if ok:
            self.lbl9.setNum(num)
            self.lbl9.adjustSize()

    def signal1(self):
        if not self.groupbox1.isChecked():
            self.chk1.setCheckState(False)

    def signal2(self):
        if not self.groupbox2.isChecked():
            self.chk2.setCheckState(False)

    def signal3(self):
        if self.groupbox3.isChecked():
            dlg = QInputDialog(self)
            dlg.setInputMode(QInputDialog.TextInput)
            dlg.setWindowTitle('Enter Password')
            dlg.setLabelText("Made by BigleaderTeam")
            dlg.resize(700, 100)
            ok = dlg.exec_()
            text = dlg.textValue()
            if ok:
                if not str(text) == 'rlacksdlf':
                    self.groupbox3.setChecked(False)
            else:
                self.groupbox3.setChecked(False)

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
        detect.run(
            weights=self.lbl1.text(),
            source=self.lbl2.text(),
            imgsz=[int(self.lbl3.text()), int(self.lbl3.text())],
            conf_thres=float(int(self.lbl4.text()[:-1])/100),
            iou_thres=float(int(self.lbl5.text()[:-1])/100),
            max_det=int(self.lbl6.text()),
            device='',
            view_img=self.chk4.isChecked(),
            save_txt=self.groupbox2.isChecked(),
            save_conf=self.chk2.isChecked(),
            save_crop=self.chk3.isChecked(),
            nosave=not self.groupbox1.isChecked(),
            classes=None if len(li) == 0 else li,
            agnostic_nms=False,
            augment=False,
            visualize=False,
            update=False,
            project=self.lbl7.text(),
            name=self.lbl8.text(),
            exist_ok=self.chk1.isChecked(),
            line_thickness=int(self.lbl9.text()),
            hide_labels=self.chk5.isChecked(),
            hide_conf=self.chk6.isChecked(),
            half=False,
            dnn=False
        )
    # def customopt(self):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--weights', nargs='+', type=str, default=self.lbl1.text(), help='model path(s)')
    #     parser.add_argument('--source', type=str, default=self.lbl2.text(), help='file/dir/URL/glob, 0 for webcam')
    #     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[int(self.lbl3.text())], help='inference size h,w')
    #     parser.add_argument('--conf-thres', type=float, default=float(self.lbl4.text()), help='confidence threshold')
    #     parser.add_argument('--iou-thres', type=float, default=float(self.lbl5.text()), help='NMS IoU threshold')
    #     parser.add_argument('--max-det', type=int, default=int(self.lbl6.text()), help='maximum detections per image')
    #     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #     parser.add_argument('--view-img', default=self.chk4.isChecked(), help='show results')
    #     parser.add_argument('--save-txt', default=self.groupbox2.isChecked(), help='save results to *.txt')
    #     parser.add_argument('--save-conf', default=self.chk2.isChecked(), help='save confidences in --save-txt labels')
    #     parser.add_argument('--save-crop', default=self.chk3.isChecked(), help='save cropped prediction boxes')
    #     parser.add_argument('--nosave', default=not self.groupbox1.isChecked(), help='do not save images/videos')
    #     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    #     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    #     parser.add_argument('--augment', action='store_true', help='augmented inference')
    #     parser.add_argument('--visualize', action='store_true', help='visualize features')
    #     parser.add_argument('--update', action='store_true', help='update all models')
    #     parser.add_argument('--project', default=f'{self.lbl7.text()} / runs/detect', help='save results to project/name')
    #     parser.add_argument('--name', default=self.lbl8.text(), help='save results to project/name')
    #     parser.add_argument('--exist-ok', default=self.chk1.isChecked(), help='existing project/name ok, do not increment')
    #     parser.add_argument('--line-thickness', default=int(self.lbl9.text()), type=int, help='bounding box thickness (pixels)')
    #     parser.add_argument('--hide-labels', default=self.chk5.isChecked(), action='store_true', help='hide labels')
    #     parser.add_argument('--hide-conf', default=self.chk6.isChecked(), action='store_true', help='hide confidences')
    #     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    #     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    #     opt = parser.parse_args()
    #     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #     return opt


class GroupBox(QGroupBox):
    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionGroupBox()
        self.initStyleOption(option)
        if self.isCheckable():
            option.state &= ~QStyle.State_Off & ~QStyle.State_On
            option.state |= (
                QStyle.State_Off
                if self.isChecked()
                else QStyle.State_On
            )
        painter.drawComplexControl(QStyle.CC_GroupBox, option)


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
