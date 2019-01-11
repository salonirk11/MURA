import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

PATH="MURA-v1.1/"
train_imgs_path=pd.read_csv(PATH+'train_image_paths.csv', header=None)
train_labels=pd.read_csv(PATH+'train_labeled_studies.csv', header=None)
test_imgs_path=pd.read_csv(PATH+'valid_image_paths.csv', header=None)
test_labels=pd.read_csv(PATH+'valid_labeled_studies.csv', header=None)



def thresh(img, val=40):
    ret,img_ = cv2.threshold(img,val,255,cv2.THRESH_TOZERO)
    return img_

def enhance(alpha, beta, image):
    new_image = np.zeros(image.shape, image.dtype)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)
    return new_image

def tone_scale(img):
    dst = cv2.equalizeHist(img)
    return dst

def unsharp(img):
    gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
    return unsharp_image


def normalise(img):
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img


def pad(img):
    delta_w = 224 - img.shape[1]
    delta_h = 224 - img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


def resize(img):
    if img.shape[0]>img.shape[1]:
        y=int((224/img.shape[0])*img.shape[1])
        im=cv2.resize(img,(y, 224), interpolation = cv2.INTER_LINEAR)
        new_im=pad(im)
    elif img.shape[0]<img.shape[1]:
        y=int((224/img.shape[1])*img.shape[0])
        im=cv2.resize(img,(224, y), interpolation = cv2.INTER_LINEAR)
        new_im=pad(im)
    else:
        new_im=cv2.resize(img,(224, 224), interpolation = cv2.INTER_LINEAR)
    return new_im

for path in test_imgs_path.values:
    img=cv2.imread(path[0])
    img=resize(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=enhance(1, 20, img)
    img=tone_scale(img)

    img=unsharp(img)

    p=path[0]
    p=p.split("/")
    p[0]='MURA-v1.2'

    directory=p[:5]
    directory='/'.join(directory)

    p=p[:6]
    p='/'.join(p)

    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(p, img)
    print(p)
