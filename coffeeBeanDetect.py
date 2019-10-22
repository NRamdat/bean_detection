#from pypylon import pylon
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

# Read images
def readImg(path):
    img = cv.imread(path)
    width, height = img.shape[:2]
    im = np.copy(img)
    return im, width, height

# Transformations
def transform(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Blur on a grayscale and get the edges
    blur = cv.bilateralFilter(gray, 7, 50, 50)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    canny = cv.Canny(blur, 35, 40)
    kernel = np.ones((5,5),np.uint8) # kernel to dilate the edges
    canny = cv.dilate(canny,kernel,iterations = 1)
    return hsv, canny

def fillInBeans(canny, thresh, maxval):
    # Fill in contours
    th, im_th = cv.threshold(canny, 0,1,cv.THRESH_BINARY)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(im_floodfill, mask, (0,0), 255)
    # Thresh to find less contours
    ret, im_th = cv.threshold(im_floodfill, thresh ,maxval, 1)
    return im_th

def getCnts(img):
    _, cnts, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE )
    return cnts

def displayImage(w_name, image, wait=False):
    width, height = image.shape[0:2]
    tmp = cv.resize(image, ((int)(height/3)+1, (int)(width/3)+1))
    cv.imshow(w_name, tmp)
    if(wait):
        cv.waitKey()


def getCntPoints(cnt, width, height):
    blank = np.zeros((width, height, 3), np.uint8)
    blank = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
    cv.drawContours(blank, cnt, -1, 255, 35)
    points_image = fillGaps(blank)
    points_image = cv.dilate(points_image,(25,25),iterations = 1)
    points_image = fillInBeans(points_image, 0, 255)
    points = np.where(points_image==255) # return index where pixel is 255 (white)
    return points

def fillGaps(tmp):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE,(16,16))
    out_img = cv.morphologyEx(tmp, cv.MORPH_CLOSE, k)
    return out_img


# Coffeebean color based on HSV Color Picker
def color(hsv, color):
    if(color == 'brown'):
        low = np.array([5, 61, 60])
        high = np.array([26 ,130,140])
        brown = cv.inRange(hsv, low, high)
        brown_draw = brown.copy()
        return brown, 'brown'
    if(color == 'yellow'):
        yellowBeanLow = np.array([5,151,145])
        yellowBeanHigh = np.array([25,171,225])
        yellow = cv.inRange(hsv, yellowBeanLow, yellowBeanHigh)
        yellow_draw = yellow.copy()
        return yellow, 'yellow'

    if(color == 'kidney'):
        kidneyLow = np.array([-6,109,76])
        kidneyHigh = np.array([14,129,156])
        kidney = cv.inRange(hsv, kidneyLow, kidneyHigh)
        kidney_draw = kidney.copy()
        return kidney, 'kidney'

def algoritme(im, cnts, color, width, height, colorName):
    count = 0
    c = cv.countNonZero(color) # number of 'color' pixels
    print("Amount brown pixels: ", c)
    #tmp = np.zeros((h,w,3), np.uint8)
    for cnt in cnts:
        # find middle of contour
        M = cv.moments(cnt)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        perimeter = cv.arcLength(cnt,True)

        if(cv.contourArea(cnt) > 40000): #and perimeter < 1040):
            print(count)
            count += 1
            print(cv.contourArea(cnt))
            perimeter = cv.arcLength(cnt,True)
            print("perimeter: ", perimeter) # Perimeter of contour

            bean_coords = getCntPoints(cnt, width, height) # coordinates within a contour, which are white pixels
            num_of_white_points = 0
            s = len(bean_coords[0])
            for i in range(0, s):
                if(color[bean_coords[0][i], bean_coords[1][i]]):
                    num_of_white_points += 1
            print("num of white points in brown image:" + str(num_of_white_points))
            if(num_of_white_points > 19000):
                cv.drawContours(im, [cnt], -1, 255, 5)
                cv.circle(im, (cX, cY), 7, (0, 0,255), -1)
                if(colorName == 'brown'):
                    cv.putText(im, "It's almost Coffee", (cX-20, cY-20),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if(colorName == 'yellow'):
                    cv.putText(im, "Huh what is this bean?", (cX-20, cY-20),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                if(colorName == 'kidney'):
                    cv.putText(im, "Kidney!!", (cX-20, cY-20),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                displayImage("test", im, wait=False)

def main():
    img = 'Bonen_110k_diaf_8_gain_0.bmp'
    #img = 'Beans5.png'
    new_img, width, height = readImg(img)
    hsv, canny = transform(new_img)
    filledBeans = fillInBeans(canny, 100, 200)
    cnts = getCnts(filledBeans)
    brown, colorName = color(hsv, 'brown')
    #yellow, colorName = color(hsv, 'yellow')
    #kidney, colorName = color(hsv, 'kidney')
    algoritme(new_img, cnts, brown, width, height, colorName) # <- change var brown to yellow or kidney e.i.
    #algoritme(new_img, cnts, yellow, width, height, colorName) # <- change var brown to yellow or kidney e.i.
    #algoritme(new_img, cnts, kidney, width, height, colorName) # <- change var brown to yellow or kidney e.i.


t = time.process_time()
main()
elapsed_time = time.process_time() - t
cv.waitKey()
print(elapsed_time, " seconds run")
cv.destroyAllWindows
