import abc
import math
import random
import numpy as np
from copy import copy

import cv2

class Filter:
    @abc.abstractmethod
    def transform(self, image: np.array):
        pass

class InversionFilter(Filter):
    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, c = image.shape
        for j in range(w):
            for i in range(h):
                for k in range(c):
                    transformed_image[j][i][k] = 255 - image[j][i][k]
        return transformed_image

class GrayScaleFilter(Filter):
    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, c = image.shape
        for j in range(w):
            for i in range(h):
                b, g, r = image[j][i]
                intensity = int(0.299 * r + 0.587 * g + 0.114 * b)
                for k in range(c):
                    transformed_image[j][i][k] = intensity
        return transformed_image

class SepiaFilter(Filter):
    def transform(self, image: np.array):
        k = 5
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                b, g, r = image[j][i]
                intensity = int(0.299 * r + 0.587 * g + 0.114 * b)
                b = np.clip(intensity - int(1 * k), 0, 255)
                g = np.clip(intensity + int(0.5 * k), 0, 255)
                r = np.clip(intensity + int(2 * k), 0, 255)
                transformed_image[j][i] = np.array([b, g, r], dtype=int)
        return transformed_image

class BrightnessFilter(Filter):
    def transform(self, image: np.array):
        k = 50
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                b, g, r = image[j][i]
                b = np.clip(b + k, 0, 255)
                g = np.clip(g + k, 0, 255)
                r = np.clip(r + k, 0, 255)
                transformed_image[j][i] = np.array([b, g, r], dtype=int)
        return transformed_image

class GrayWorldFilter(Filter):
    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape

        B, G, R = cv2.split(image)
        b_avg = np.average(B)
        g_avg = np.average(G)
        r_avg = np.average(R)
        avg = np.average([b_avg, g_avg, r_avg])

        for j in range(w):
            for i in range(h):
                b, g, r = image[j][i]
                b = int(b * avg / b_avg)
                g = int(g * avg / g_avg)
                r = int(r * avg / r_avg)
                transformed_image[j][i] = np.array([b, g, r], dtype=int)
        return transformed_image

class LinearCorrection(Filter):
    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape

        B, G, R = cv2.split(image)
        bmin, bmax = np.amin(B), np.amax(B)
        gmin, gmax = np.amin(G), np.amax(G)
        rmin, rmax = np.amin(R), np.amax(R)

        for j in range(w):
            for i in range(h):
                b, g, r = image[j][i]
                b = self.__f(b, bmin, bmax)
                g = self.__f(g, gmin, gmax)
                r = self.__f(r, rmin, rmax)
                transformed_image[j][i] = np.array([b, g, r], dtype=int)
        return transformed_image

    def __f(self, y: int, ymin: int, ymax: int):
        return int(255 * (y - ymin) / (ymax - ymin))

class GlassFilter(Filter):
    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                k = np.clip(j - int(10 * (random.random() - 0.5)), 0, w - 1)
                l = np.clip(i - int(10 * (random.random() - 0.5)), 0, h - 1)
                transformed_image[k][l] = copy(image[j][i])
        return transformed_image

class WavesFilter(Filter):
    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                k = np.clip(int(j + 20 * math.sin(2 * math.pi * j / 60)), 0, w - 1)
                transformed_image[j][i] = copy(image[k][i])
        return transformed_image

class MatrixFilter(Filter):
    def __init__(self, kernel: np.array):
        self._kernel = kernel

    def _calculate_pixel_color(self, image: np.array, x: int, y: int):
        w_filter, h_filter = self._kernel.shape
        x_radius = int(w_filter / 2)
        y_radius = int(h_filter / 2)
        new_color = np.zeros(shape=(3,), dtype=int)
        w, h, c = image.shape
        for j in range(-x_radius, x_radius + 1):
            for i in range(-y_radius, y_radius + 1):
                index_x = np.clip(x + j, 0, w - 1)
                index_y = np.clip(y + i, 0, h - 1)
                curr_color = image[index_x][index_y]
                for c in range(3):
                    new_color[c] += curr_color[c] * self._kernel[j + x_radius][i + y_radius]
        for c in range(3):
            new_color[c] = np.clip(new_color[c], 0, 255)
        return new_color

class BlurFilter(MatrixFilter):
    def __init__(self):
        super().__init__(np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]], dtype=float))

    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                transformed_image[j][i] = self._calculate_pixel_color(image, j, i)
        return transformed_image

class GaussianFilter(MatrixFilter):
    def __init__(self):
        super().__init__(self.__create_gaussian_kernel(3, 2))

    def __create_gaussian_kernel(self, radius: int, sigma: float):
        size = int(2 * radius + 1)
        kernel = np.zeros(shape=(size, size,), dtype=float)
        norm = 0.0
        for j in range(-radius, radius + 1):
            for i in range(-radius, radius + 1):
                kernel[j + radius][i + radius] = float(math.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2)))
                norm += kernel[j + radius][i + radius]
        kernel = kernel / norm
        return kernel

    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                transformed_image[j][i] = self._calculate_pixel_color(image, j, i)
        return transformed_image

class StampingFilter(MatrixFilter):
    def __init__(self):
        super().__init__(np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=float))

    def transform(self, image: np.array):
        filter = GrayScaleFilter()
        transformed_image = filter.transform(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                transformed_image[j][i] = self._calculate_pixel_color(image, j, i)
                transformed_image[j][i] = transformed_image[j][i] + 255
                for k in range(3):
                    transformed_image[j][i][k] = int(transformed_image[j][i][k] / 2)
        return transformed_image

class MotionBlurFilter(MatrixFilter):
    def __init__(self):
        super().__init__(np.array([[1/5, 0, 0, 0, 0], [0, 1/5, 0, 0, 0],
            [0, 0, 1/5, 0, 0], [0, 0, 0, 1/5, 0], [0, 0, 0, 0, 1/5]], dtype=float))

    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                transformed_image[j][i] = self._calculate_pixel_color(image, j, i)
        return transformed_image

class MedianFilter(Filter):
    def __calculate_pixel_color(self, image: np.array, x: int, y: int, radius: int):
        w, h, _ = image.shape
        b_arr = []
        g_arr = []
        r_arr = []
        for j in range(-radius, radius + 1):
            for i in range(-radius, radius + 1):
                index_x = np.clip(x + j, 0, w - 1)
                index_y = np.clip(y + i, 0, h - 1)
                b_arr.append(image[index_x][index_y][0])
                g_arr.append(image[index_x][index_y][1])
                r_arr.append(image[index_x][index_y][2])
        new_color = np.array([np.median(b_arr), np.median(g_arr), np.median(r_arr)], dtype=int)
        return new_color

    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                transformed_image[j][i] = self.__calculate_pixel_color(image, j, i, 3)
        return transformed_image

class Dilation(Filter):
    def __init__(self):
        self.__mask = self.__get_mask()

    def __get_mask(self):
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        return mask

    def __calculate_pixel_color(self, image: np.array, x: int, y: int):
        w_filter, h_filter = self.__mask.shape
        x_radius = int(w_filter / 2)
        y_radius = int(h_filter / 2)

        w, h, _ = image.shape
        b_max = 0
        g_max = 0
        r_max = 0

        for j in range(-x_radius, x_radius + 1):
            for i in range(-y_radius, y_radius + 1):
                index_x = np.clip(x + j, 0, w - 1)
                index_y = np.clip(y + i, 0, h - 1)
                if self.__mask[j][i]:
                    b_max = max(b_max, image[index_x][index_y][0])
                    g_max = max(g_max, image[index_x][index_y][1])
                    r_max = max(r_max, image[index_x][index_y][2])
        new_color = np.array([b_max, g_max, r_max], dtype=int)
        return new_color

    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                transformed_image[j][i] = self.__calculate_pixel_color(image, j, i)
        return transformed_image

class Erosion(Filter):
    def __init__(self):
        self.__mask = self.__get_mask()

    def __get_mask(self):
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        return mask

    def __calculate_pixel_color(self, image: np.array, x: int, y: int):
        w_filter, h_filter = self.__mask.shape
        x_radius = int(w_filter / 2)
        y_radius = int(h_filter / 2)

        w, h, _ = image.shape
        b_min = 255
        g_min = 255
        r_min = 255

        for j in range(-x_radius, x_radius + 1):
            for i in range(-y_radius, y_radius + 1):
                index_x = np.clip(x + j, 0, w - 1)
                index_y = np.clip(y + i, 0, h - 1)
                if self.__mask[j][i]:
                    b_min = min(b_min, image[index_x][index_y][0])
                    g_min = min(g_min, image[index_x][index_y][1])
                    r_min = min(r_min, image[index_x][index_y][2])
        new_color = np.array([b_min, g_min, r_min], dtype=int)
        return new_color

    def transform(self, image: np.array):
        transformed_image = copy(image)
        w, h, _ = image.shape
        for j in range(w):
            for i in range(h):
                transformed_image[j][i] = self.__calculate_pixel_color(image, j, i)
        return transformed_image

class Open(Filter):
    def transform(self, image: np.array):
        dilation = Dilation()
        erosion = Erosion()
        return erosion.transform(dilation.transform(image))

class Close(Filter):
    def transform(self, image: np.array):
        dilation = Dilation()
        erosion = Erosion()
        return dilation.transform(erosion.transform(image))

class Grad(Filter):
    def transform(self, image: np.array):
        dilation = Dilation()
        erosion = Erosion()
        return dilation.transform(image) - erosion.transform(image)
