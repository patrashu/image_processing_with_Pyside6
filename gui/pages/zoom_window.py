import sys
import cv2

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QApplication, QMainWindow, QMdiSubWindow, QLineEdit
from PySide6.QtGui import QPixmap, QImage

# QmainWindow를 상속 받아 앱의 Main Window를 커스텀 합시다!
class ZoomWindow(QMdiSubWindow):
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
        if len(self.offsets) == 2:
            from gui.pages.ui_pages import Ui_application_pages
            x1, y1 = self.offsets[-2]
            x2, y2 = self.offsets[-1]
            self.copy_image = self.copy_image[y1:y2, x1:x2, :]
            cv2.imwrite('result.jpg', self.copy_image)
            self.close()