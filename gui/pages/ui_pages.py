# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'pagesIujefo.ui'
##
## Created by: Qt User Interface Compiler version 6.1.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from qt_core import *
import cv2
import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized, TracedModel
from gui.pages.zoom_window import ZoomWindow
from gui.pages.warp_window import WarpWindow
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
efficientnet.eval().to(device)

class Ui_application_pages(object):
    def setupUi(self, application_pages):
        self.temp = None
        self.offsets = []
        if not application_pages.objectName():
            application_pages.setObjectName(u"application_pages")
        application_pages.resize(1400, 960)
        
        #Page 1
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")
        
        self.main_layout_1 = QVBoxLayout(self.page_1)
        
        self.sub_layout_1 = QHBoxLayout()
        self.main_layout_1.addLayout(self.sub_layout_1)
        
        self.frame = QFrame(self.page_1)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(120, 960))
        self.frame.setMaximumSize(QSize(120, 960))
        # self.frame.setFrameShape(QFrame.StyledPanel)
        # self.frame.setFrameShadow(QFrame.Raised)

        self.buttonLayout = QVBoxLayout(self.frame)
        self.buttonLayout.setSpacing(4)
        self.buttonLayout.setObjectName(u"buttonLayout")
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        
        self.image_open_button = QPushButton("이미지\n열기")
        self.image_open_button.setObjectName(u"image_open_button")
        self.image_open_button.setMinimumSize(QSize(60, 50))
        self.image_open_button.setMaximumSize(QSize(60, 50))
        self.image_open_button.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.image_open_button.clicked.connect(self.show_file_dialog)
        self.buttonLayout.addWidget(self.image_open_button)
        
        self.image_clear = QPushButton("새로고침")
        self.image_clear.setObjectName(u"image_clear")
        self.image_clear.setMinimumSize(QSize(60, 50))
        self.image_clear.setMaximumSize(QSize(60, 50))
        self.image_clear.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")

        self.image_clear.clicked.connect(self.clear_label)
        self.buttonLayout.addWidget(self.image_clear)
        
        self.image_flip = QPushButton("좌우반전")
        self.image_flip.setObjectName(u"image_flip")
        self.image_flip.setMinimumSize(QSize(60, 50))
        self.image_flip.setMaximumSize(QSize(60, 50))
        self.image_flip.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.image_flip.clicked.connect(self.flip_image)
        self.buttonLayout.addWidget(self.image_flip)
        
        self.image_normalization = QPushButton("정규화")
        self.image_normalization.setMinimumSize(QSize(60, 50))
        self.image_normalization.setMaximumSize(QSize(60, 50))
        self.image_normalization.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.image_normalization.clicked.connect(self.normalization_image)
        self.buttonLayout.addWidget(self.image_normalization)

        self.image_sharpning = QPushButton("이미지\n선명화")
        self.image_sharpning.setObjectName(u"image_sharping")
        self.image_sharpning.setMinimumSize(QSize(60, 50))
        self.image_sharpning.setMaximumSize(QSize(60, 50))
        self.image_sharpning.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")

        self.image_sharpning.clicked.connect(self.sharpning)
        self.buttonLayout.addWidget(self.image_sharpning)

        self.image_boundary = QPushButton("경계선\n검출")
        self.image_boundary.setObjectName(u"image_boundary")
        self.image_boundary.setMinimumSize(QSize(60, 50))
        self.image_boundary.setMaximumSize(QSize(60, 50))
        self.image_boundary.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")

        self.image_boundary.clicked.connect(self.extract_boundary)
        self.buttonLayout.addWidget(self.image_boundary)

        self.warp_image = QPushButton('이미지\n워핑')
        self.warp_image.setObjectName(u'warping_image')
        self.warp_image.setMinimumSize(QSize(60, 50))
        self.warp_image.setMaximumSize(QSize(60, 50))
        self.warp_image.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.warp_image.clicked.connect(self.warping)
        self.buttonLayout.addWidget(self.warp_image)

        self.zoom_image = QPushButton("이미지\n확대")
        self.zoom_image.setObjectName(u'zoom_image')
        self.zoom_image.setMinimumSize(QSize(60, 50))
        self.zoom_image.setMaximumSize(QSize(60, 50))
        self.zoom_image.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.zoom_image.clicked.connect(self.image_zoom)
        self.buttonLayout.addWidget(self.zoom_image)
        
        self.result_apply = QPushButton("결과적용")
        self.result_apply.setObjectName(u'result_apply')
        self.result_apply.setMinimumSize(QSize(60, 50))
        self.result_apply.setMaximumSize(QSize(60, 50))
        self.result_apply.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.result_apply.clicked.connect(self.apply_result)
        self.buttonLayout.addWidget(self.result_apply)

        self.sub_layout_1.addWidget(self.frame)
        self.origin = QLabel()
        self.origin.setFixedSize(640, 960)
        self.sub_layout_1.addWidget(self.origin)
        
        self.change = QLabel()
        self.change.setFixedSize(640, 960)
        self.sub_layout_1.addWidget(self.change)
        application_pages.addWidget(self.page_1)
        
        #Page 2
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.main_layout_2 = QVBoxLayout(self.page_2)
        self.main_layout_2.setObjectName(u"main_layout_2")
        
        self.sub_layout_2 = QHBoxLayout()
        self.main_layout_2.addLayout(self.sub_layout_2)
        
        self.frame_2 = QFrame(self.page_2)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMinimumSize(QSize(120, 960))
        self.frame_2.setMaximumSize(QSize(120, 960))
        
        self.buttonLayout_2 = QVBoxLayout(self.frame_2)
        self.buttonLayout_2.setSpacing(4)
        self.buttonLayout_2.setObjectName(u"buttonLayout_2")
        self.buttonLayout_2.setContentsMargins(0, 0, 0, 0)
        
        self.image_open_button_2 = QPushButton("이미지\n열기")
        self.image_open_button_2.setObjectName(u"image_open_button_2")
        self.image_open_button_2.setMinimumSize(QSize(60, 50))
        self.image_open_button_2.setMaximumSize(QSize(60, 50))
        self.image_open_button_2.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.image_open_button_2.clicked.connect(self.show_file_dialog_2)
        self.buttonLayout_2.addWidget(self.image_open_button_2)
        
        self.image_classification_button = QPushButton("이미지\n분류")
        self.image_classification_button.setObjectName(u"image_classification_button")
        self.image_classification_button.setMinimumSize(QSize(60, 50))
        self.image_classification_button.setMaximumSize(QSize(60, 50))
        self.image_classification_button.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.image_classification_button.clicked.connect(self.image_classification)
        self.buttonLayout_2.addWidget(self.image_classification_button)
        
        self.sub_layout_2.addWidget(self.frame_2)
        self.origin_2 = QLabel()
        self.origin_2.setFixedSize(640, 960)
        self.sub_layout_2.addWidget(self.origin_2)
        
        self.label_layout = QVBoxLayout()
        self.result = QLabel('EfficientNet-B0 Result')
        self.result.setFixedSize(640, 200)
        self.result.setStyleSheet("font-size: 40pt\"Segoe UI\"; color: rgb(70,130,180);")
        self.result.setAlignment(Qt.AlignCenter)
        self.label_layout.addWidget(self.result)
        self.result_1 = QLabel()
        self.result_1.setFixedSize(640, 200)
        self.result_1.setAlignment(Qt.AlignCenter)
        self.result_1.setStyleSheet("font-size: 20pt; color: rgb(0,191,255);")
        self.label_layout.addWidget(self.result_1)
        self.result_2 = QLabel()
        self.result_2.setFixedSize(640, 200)
        self.result_2.setAlignment(Qt.AlignCenter)
        self.result_2.setStyleSheet("font-size: 20pt; color: rgb(176,224,230);")
        self.label_layout.addWidget(self.result_2)
        self.result_3 = QLabel()
        self.result_3.setFixedSize(640, 200)
        self.result_3.setAlignment(Qt.AlignCenter)
        self.result_3.setStyleSheet("font-size: 20pt; color: rgb(176,224,230);")
        self.label_layout.addWidget(self.result_3)
        
        self.sub_layout_2.addLayout(self.label_layout)

        application_pages.addWidget(self.page_2)
        
        #Page 3
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        
        self.main_layout_3 = QVBoxLayout(self.page_3)
        self.main_layout_3.setObjectName(u"main_layout_3")
        
        self.sub_layout_3 = QHBoxLayout()
        self.main_layout_3.addLayout(self.sub_layout_3)
        
        self.frame_3 = QFrame(self.page_3)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setMinimumSize(QSize(120, 960))
        self.frame_3.setMaximumSize(QSize(120, 960))
        
        self.buttonLayout_3 = QVBoxLayout(self.frame_3)
        self.buttonLayout_3.setSpacing(4)
        self.buttonLayout_3.setObjectName(u"buttonLayout_3")
        self.buttonLayout_3.setContentsMargins(0, 0, 0, 0)
        
        self.image_open_button_3 = QPushButton("이미지\n열기")
        self.image_open_button_3.setObjectName(u"image_open_button_3")
        self.image_open_button_3.setMinimumSize(QSize(60, 50))
        self.image_open_button_3.setMaximumSize(QSize(60, 50))
        self.image_open_button_3.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.image_open_button_3.clicked.connect(self.show_file_dialog_3)
        self.buttonLayout_3.addWidget(self.image_open_button_3)
        
        self.object_detection_button = QPushButton("객체인식")
        self.object_detection_button.setObjectName(u"object_detection_button")
        self.object_detection_button.setMinimumSize(QSize(60, 50))
        self.object_detection_button.setMaximumSize(QSize(60, 50))
        self.object_detection_button.setStyleSheet(u"QPushButton {\n"
"	background-color: rgb(68, 71, 90);\n"
"	padding: 8px;\n"
"	border: 2px solid #c3ccdf;\n"
"	color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"   font-size:10px;\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:pressed {\n"
"	background-color: rgb(255, 0, 127);\n"
"}")
        self.object_detection_button.clicked.connect(self.object_detection)
        self.buttonLayout_3.addWidget(self.object_detection_button)
        
        self.sub_layout_3.addWidget(self.frame_3)
        
        self.origin_3 = QLabel()
        self.origin_3.setFixedSize(640, 960)
        self.sub_layout_3.addWidget(self.origin_3)
        
        self.change_3 = QLabel()
        self.change_3.setFixedSize(640, 960)
        self.sub_layout_3.addWidget(self.change_3)

        application_pages.addWidget(self.page_3)

        self.retranslateUi(application_pages)

        QMetaObject.connectSlotsByName(application_pages)

    def retranslateUi(self, application_pages):
        application_pages.setWindowTitle(QCoreApplication.translate("application_pages", u"StackedWidget", None))
    
    def show_file_dialog(self):
        file_name = QFileDialog.getOpenFileName(self.page_1, "이미지 열기", "./")
        self.image = cv2.imread(file_name[0])
        self.input_path = file_name[0]
        self.image = cv2.resize(self.image, (640, 960))
        h, w, _ = self.image.shape
        bytes_per_line = 3 * w
        
        image = QImage(
            self.image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image)
        self.origin.setPixmap(pixmap)
        
    def show_file_dialog_2(self):
        file_name_2 = QFileDialog.getOpenFileName(self.page_2, "이미지 열기", "./")
        self.image_2 = cv2.imread(file_name_2[0])
        self.image_2 = cv2.resize(self.image_2, (640, 960))
        h_2, w_2, _ = self.image_2.shape
        bytes_per_line_2 = 3 * w_2
        
        image_2 = QImage(
            self.image_2.data, w_2, h_2, bytes_per_line_2, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap_2 = QPixmap(image_2)
        self.origin_2.setPixmap(pixmap_2)
        
    def show_file_dialog_3(self):
        file_name_3 = QFileDialog.getOpenFileName(self.page_3, "이미지 열기", "./")
        self.image_3 = cv2.imread(file_name_3[0])
        self.image_3 = cv2.resize(self.image_3, (640, 640))
        h_3, w_3, _ = self.image_3.shape
        bytes_per_line_3 = 3 * w_3
        
        image_3 = QImage(
            self.image_3.data, w_3, h_3, bytes_per_line_3, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap_3 = QPixmap(image_3)
        self.origin_3.setPixmap(pixmap_3)
    
    #좌우반전
    def flip_image(self):
        flip_image = cv2.flip(self.image, 1)
        h, w, _ = flip_image.shape
        bytes_per_line = 3 * w
        flip_image = QImage(
            flip_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(flip_image)
        self.change.setPixmap(pixmap)
        
    #정규화
    def normalization_image(self):
        normalization = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)
        h, w, _ = normalization.shape
        bytes_per_line = 3 * w
        normalization = QImage(
            normalization.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(normalization)
        self.change.setPixmap(pixmap)
        
    #새로고침
    def clear_label(self):
        self.temp = None
        self.change.clear()
        
    #선명화
    def sharpning(self):
        sharp_image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)
        kernel = np.array(
                    [[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]]
                )
        sharp_image = cv2.filter2D(sharp_image, -1, kernel)
        h, w, _ = sharp_image.shape
        bytes_per_line = 3 * w
        sharp_image = QImage(
            sharp_image, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(sharp_image)
        self.change.setPixmap(pixmap)

    ## 경계선 검출
    def extract_boundary(self):
        boundary_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        median_intensity = np.median(boundary_image)
        lower_threshold = 100
        upper_threshold = 200
        boundary_image = cv2.Canny(boundary_image, lower_threshold, upper_threshold)
        h, w = boundary_image.shape
        bytes_per_line = w
        boundary_image = QImage(
            boundary_image, w, h, bytes_per_line, QImage.Format_Grayscale8
        ).rgbSwapped()

        pixmap = QPixmap(boundary_image)
        self.change.setPixmap(pixmap)

    ## 이미지 워핑
    def warping(self):
        msg = QMessageBox()
        msg.setText("마우스로 워핑할 꼭짓점 총 8개를 클릭해주세요")
        result = msg.exec_()
        if result == QMessageBox.Cancel:
            self.send_valve_popup_signal.emit(False)
        
        subwindow = WarpWindow(self.input_path)
        subwindow.show()

    ## 이미지 확대   
    def image_zoom(self):
        msg = QMessageBox()
        msg.setText("마우스로 확대할 두 점을 클릭해주세요")
        result = msg.exec_()
        if result == QMessageBox.Cancel:
            self.send_valve_popup_signal.emit(False)
        
        subwindow = ZoomWindow(self.input_path)
        subwindow.show()

    def apply_result(self):
        res_image = cv2.imread("result.jpg")
        res_image = cv2.resize(res_image, (640, 960))

        h, w, _ = res_image.shape
        bytes_per_line = 3 * w

        res_image = QImage(
            res_image, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        pixmap = QPixmap(res_image)
        self.change.setPixmap(pixmap)
        os.remove('result.jpg')
        
    def image_classification(self):
        classification_image = self.image_2.copy()
        classification_image = cv2.cvtColor(classification_image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        classification_image = transform(classification_image)
        classification_image = classification_image.unsqueeze(0)
        batch = classification_image.to(device)
        with torch.no_grad():
            output = torch.nn.functional.softmax(efficientnet(batch), dim=1)
        results = utils.pick_n_best(predictions=output, n=5)
        print('Top 1 : ' + str(results[0][0][0]))
        self.result_1.setText('Top 1 : ' + str(results[0][0][0]))
        self.result_2.setText('Top 2 : ' + str(results[0][1][0]))
        self.result_3.setText('Top 3 : ' + str(results[0][2][0]))

    def object_detection(self):
        detection_image = self.image_3.copy()
        self.batch = detection_image
        self.detect()
        
    def detect(self):
        source = self.batch
        weights = 'yolov7.pt'
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = 640
        imgsz = check_img_size(imgsz, s=stride)
        half = device.type != 'cpu'
        model = TracedModel(model, device, 640)
        img0 = cv2.flip(source, 1)
        img = letterbox(img0, (640, 640), stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        
        # for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        t3 = time_synchronized()
        save_dir = ''
        path = ''
        for i, det in enumerate(pred):  # detections per image
            s = ''
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                img0 = cv2.flip(img0, 1)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

        h, w, _ = img0.shape
        bytes_per_line = 3 * w
        detect_image = QImage(
            img0.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(detect_image)
        self.change_3.setPixmap(pixmap)
                
    def mouseClickEvent(self, event):
        self.offsets.append((event.x(), event.y()))