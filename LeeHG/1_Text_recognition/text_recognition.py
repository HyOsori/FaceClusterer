import sys
from pytesseract import *
from PIL import Image
import cv2 as cv
import numpy as np


''' blur canny 후 recog
    => 정확도 실패

def get_image(filename):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img,(3,3),0)
    cv.imwrite('blur.jpg',blur)
    canny = cv.Canny(blur, 100, 200)
    cv.imwrite('canny.jpg',canny)

    return img



    #return Image.open(filename)
    #return cv.imread(filename)

if __name__ == '__main__' :
    img = get_image('textimg.jpg')
    sys.stdout.write('=== TEXT FROM IMAGE ==='+'\n')

    sys.stdout.write(image_to_string(img, lang = "kor"))
'''

def OCR(img, lang = 'eng'):
    if lang == 'kor' :
        im = Image.open(img)
    else :
        im = Image.open(img)
    text = image_to_string(im, lang = lang)

    print(text)

OCR('jam1.jpg', lang = 'kor')