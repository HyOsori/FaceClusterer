import sys
from pytesseract import image_to_string
from PIL import Image
import cv2 as cv
import numpy as np



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