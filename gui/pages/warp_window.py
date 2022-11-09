import sys
import cv2
import numpy as np

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QApplication, QMainWindow, QMdiSubWindow, QLineEdit
from PySide6.QtGui import QPixmap, QImage

# QmainWindow를 상속 받아 앱의 Main Window를 커스텀 합시다!
class WarpWindow(QMdiSubWindow):
    def __init__(self, image_path) -> None:
        super().__init__()

        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 960))
        self.copy_image = image.copy()
        
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        image = QImage(
            image.data, w, h, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(image)
        
        
        self.label = QLabel(self)
        self.label.setFixedSize(640, 960)
        self.label.setPixmap(pixmap)

        self.setWindowTitle("Image")
        self.setFixedSize(QSize(700, 1000))

        self.label.mousePressEvent = self.mousePressEvent
        self.offsets = []

    def mousePressEvent(self, e) -> None:
        self.offsets.append((e.x(), e.y()))
        print(e.x(), e.y())
        self.check()

    def check(self):
        if len(self.offsets) == 8:
            line = ""

            for i in range(8):
                x, y = self.offsets[i]
                line += f"{x} {y} "

            line = line.split(' ')

            srcPnt = np.array([
                [line[0], line[1]], [line[2], line[3]], [line[4], line[5]], [line[6], line[7]]
            ], dtype=np.float32)

            dstPnt = np.array([
                [line[8], line[9]], [line[10], line[11]], [line[12], line[13]], [line[14], line[15]]
            ], dtype=np.float32)

            h, w, _ = self.copy_image.shape

            matrix = cv2.getPerspectiveTransform(srcPnt, dstPnt)
            dst = cv2.warpPerspective(self.copy_image, matrix, (w, h))

            cv2.imwrite('result.jpg', dst)
            self.close()