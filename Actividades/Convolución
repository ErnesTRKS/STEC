"""
By Abhisek Jana and Samrat Sahoo
code taken from https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
and https://gist.githubusercontent.com/SamratSahoo/cef04a39a4033f7bec0299a10701eb95/raw/a8d327d0e38517e63aa73e334d4e656f4e3ead0a/convolution.py
blog http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
blog https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
Modified by Ernesto Del Toro
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def processImage(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return image

def conv_helper(fragment, kernel):
    """ multiplica 2 matices y devuelve su suma"""

    f_row, f_col = fragment.shape
    k_row, k_col = kernel.shape
    result = 0.0
    for row in range(f_row):
        for col in range(f_col):
            result += fragment[row,col] *  kernel[row,col]
    return result

def convolution(image, kernel):
    """Aplica una convolucion sin padding (valida) de una dimesion
    y devuelve la matriz resultante de la operación
    """

    image_row, image_col = image.shape #asigna alto y ancho de la imagen
    kernel_row, kernel_col = kernel.shape #asigna alto y ancho del filtro

    output = np.zeros(image.shape) #matriz donde guardo el resultado

    for row in range(image_row):
        for col in range(image_col):
                output[row, col] = conv_helper(
                                    image[row:row + kernel_row,
                                    col:col + kernel_col],kernel)

    plt.imshow(output, cmap='gray')
    plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
    plt.show()

    return output
if __name__ == '__main__':

    image = processImage('Image.jpg')
    kernel = np.ones([3,3])
    output = convolution(image, kernel)
