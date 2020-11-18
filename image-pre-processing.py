import cv2
import numpy as np
import os


from PIL import Image
from resizeimage import resizeimage




path1 = "./dataset/test/duvar/"
listing = os.listdir(path1)    

for el in listing:
    tt = str(path1 + el)
    #img = cv2.imread(tt)
    img = Image.open(tt)
    '''
    #percent by which the image is resized
    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)
    # resize image
    output = cv2.resize(img, dsize)
    cv2.imwrite(tt, output)
    print("*")
    '''
    img = resizeimage.resize_contain(img, [128, 128])
    img = img.convert("RGB")
    # save image
    img.save(tt, img.format)
    print("*")