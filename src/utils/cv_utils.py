import cv2
import imageio
from utils import cv_utils

def read_image(image_path):
    """
        reads the image
        :param image_path: string image path

    """
    print(image_path)
    return cv2.imread(image_path)


def convert_gray(image):
    """
        converts bgr image to gray image
        :param image: numpy array
                      gray image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_gaussian(image):
    """
        applies a gaussian filter to the image
        :param image: numpy array
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_canny(image, low_threshold, high_threshold):
    """
            detects edges in the image
            :param image: numpy array
            :param low_threshold: int
            :param high_threshold: int
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def resize(image, shape):
    """
        returns a resize image
        :param image: numpy array
                      image which is to be resize
        :param shape: tuple with exactly two elements (width, height)
                      shape to which image has to be scaled
    """
    return cv2.resize(image, shape)


def transform_image(img, transform,device):

     
    tf_img = transform(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    tf_img = tf_img.reshape(1, 3, 448, 448)
    tf_img = tf_img.to(device)

    return tf_img

 
