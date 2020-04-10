from collections import OrderedDict

import cv2
from  PyQt5 import QtWidgets, QtGui

from filter_impl import filters

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__original_image = None
        self.__transformed_image = None
        self.__filters = OrderedDict([
                ("InversionFilter",filters.InversionFilter()), 
                ("GrayScaleFilter", filters.GrayScaleFilter()), 
                ("SepiaFilter", filters.SepiaFilter()),
                ("BrightnessFilter", filters.BrightnessFilter()),
                ("GrayWorldFilter", filters.GrayWorldFilter()),
                ("LinearCorrection", filters.LinearCorrection()),
                ("GlassFilter", filters.GlassFilter()),
                ("WavesFilter", filters.WavesFilter()),
                ("BlurFilter", filters.BlurFilter()),
                ("GaussianFilter", filters.GaussianFilter()),
                ("StampingFilter", filters.StampingFilter()),
                ("MotionBlurFilter", filters.MotionBlurFilter()),
                ("MedianFilter", filters.MedianFilter()),
                ("Dilation", filters.Dilation()),
                ("Erosion", filters.Erosion()),
                ("Open", filters.Open()),
                ("Close", filters.Close()),
                ("Grad", filters.Grad())
            ])
        self.__curr_filter = self.__filters["InversionFilter"]
        self.__grid = QtWidgets.QGridLayout(self)
        self.__init_original_image_label()
        self.__init_transformed_image_label()
        self.__init_button_open_image()
        self.__init_filter_list()
        self.__init_button_apply_filter()

    def __init_original_image_label(self):
        self.__original_image_frame = QtWidgets.QLabel()
        self.__original_image_frame.setText("Original Image will be here")
        self.__grid.addWidget(self.__original_image_frame, 0, 0)

    def __init_transformed_image_label(self):
        self.__transformed_image_frame = QtWidgets.QLabel()
        self.__transformed_image_frame.setText("Transformed Image will be here")
        self.__grid.addWidget(self.__transformed_image_frame, 0, 1)

    def __init_button_open_image(self):
        button = QtWidgets.QPushButton()
        button.setText("Open Image")
        button.clicked.connect(self.__connect_open_image)
        self.__grid.addWidget(button, 1, 0, 1, 2)

    def __connect_open_image(self):
        image_name = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", ".")[0]
        self.__original_image = cv2.imread(image_name)
        image = QtGui.QImage(self.__original_image.data, self.__original_image.shape[1], self.__original_image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.__original_image_frame.setPixmap(QtGui.QPixmap.fromImage(image))

    def __init_filter_list(self):
        filter_list = QtWidgets.QComboBox()
        filter_list.addItems(self.__filters.keys())
        filter_list.activated[str].connect(self.__connect_change_filter)
        self.__grid.addWidget(filter_list, 2, 0)

    def __connect_change_filter(self, filter_name: str):
        self.__curr_filter = self.__filters[filter_name]

    def __init_button_apply_filter(self):
        button = QtWidgets.QPushButton()
        button.setText("Apply Filter")
        button.clicked.connect(self.__connect_apply_filter)
        self.__grid.addWidget(button, 2, 1)

    def __connect_apply_filter(self, filter_name: str):
        self.__transformed_image = self.__curr_filter.transform(self.__original_image)
        image = QtGui.QImage(self.__transformed_image.data, self.__transformed_image.shape[1], self.__transformed_image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.__transformed_image_frame.setPixmap(QtGui.QPixmap.fromImage(image))

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 720)
        self.setWindowTitle("Computer Graphics. Laboratory Work 1.")
        self.__central = MainWidget(self)
        self.setCentralWidget(self.__central)
