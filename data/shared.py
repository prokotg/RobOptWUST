from cv2 import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

def calculate_mask(img):
    # convert img to grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # set a thresh
    thresh = 3
    # get threshold image
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create an empty image for contours
    img_contours = np.zeros(img.shape[:2])
    # draw the contours on the empty image
    cv2.drawContours(img_contours, contours, -1, 1, thickness=cv2.FILLED)
    # erode small spots
    img_contours = cv2.erode(img_contours, np.ones((3, 3), np.uint8), iterations=1)
    # close gaps with bigger kernel
    img_contours = cv2.dilate(img_contours, np.ones((7, 7), np.uint8), iterations=1)
    return img_contours


def set_background(foreground, background):
    background = TF.resize(background, (224, 224))
    image = TF.resize(foreground, (224, 224))
    mask = calculate_mask(np.asarray(image))
    mask = (mask == 0.0)
    image = TF.to_tensor(image)
    image[:, mask] = (background.float() / 255)[:, mask]
    return image, mask


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
