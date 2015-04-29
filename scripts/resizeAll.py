#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import os,sys

image_type = [".tif",".jpg",".png",".bmp",".pgm"]

def list_images(dir):
    files = os.listdir(dir)
    images = []
    for f in files:
        name,ext=os.path.splitext(f)
        if ext in image_type:
            images.append(f)
    return images

new_path_ext = "/resized/"

def resize_all(dir,fx=0.25,fy=0.25,dst=None,nearest=False):
    files = list_images(dir)
    output_directory = dir+new_path_ext
    try:
        os.mkdir(output_directory)
    except:
        pass
    for f in files:
        im = cv2.imread(f)
        name,ext=os.path.splitext(f)
        if not nearest:
            if fx>1 or fy>1:
                imr = cv2.resize(im,dsize=dst,fx=fx,fy=fy, interpolation=cv2.INTER_CUBIC)
            else:
                imr = cv2.resize(im,dsize=dst,fx=fx,fy=fy, interpolation=cv2.INTER_AREA)
        else:
            imr = cv2.resize(im,dsize=dst,fx=fy,fy=fy, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_directory+name+".png",imr)


NEAR = True
if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "./"
    dst = None
    ff = None
    if len(sys.argv) ==3:
        ff = int(sys.argv[2])
    if len(sys.argv) ==4:
        dst = (int(sys.argv[2]),int(sys.argv[3]))
    if dst is not None:
        resize_all(dataset,0,0,dst,NEAR)
    elif ff is not None:
        resize_all(dataset,ff,ff,nearest=NEAR)
    else:
        resize_all(dataset,nearest=NEAR)
