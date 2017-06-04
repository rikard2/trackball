import numpy as np
import cv2
import random

cap = cv2.VideoCapture('rally2.mpg')
fgbg = cv2.BackgroundSubtractorMOG(1, 1, 0.9, 25)

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    b,g,r = cv2.split(frame)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    edges = cv2.Canny(blur, 10, 80)
    #gmask = fgbg.apply(frame)
    #cv2.rectangle(frame, (100, 100), (50, 50), (0,255,0), 1)

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print('hierarchy', contours, hierarchy)
    #cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    nr = 0
    for c in contours:
        rect = cv2.boundingRect(c)
        minrect = cv2.minAreaRect(c)
        size = minrect[1][0] * minrect[1][1]
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        color = ( 0, 255, 0 )
        print('contour', rect)
        if (size > 500):
            cv2.drawContours(frame, contours[nr], -1, color, 3)
        nr = nr + 1
    #gray = cv2.cvtColor(frame, cv2.CV_BGR2GRAY)
    #threshold(gray, gray,30, 255,THRESH_BINARY_INV) //Threshold the gray
    #omg, lol = cv2.threshold(hsv_img,150,160,cv2.THRESH_TOZERO)
    #if ret == True:
    #    img = cv2.rectangle(frame, (0, 0), (100, 100), 255,2)
    #    cv2.imshow('frame', img)
    #mask = cv2.inRange(hsv_img, lower_purple, upper_purple)
    #contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('edges', edges)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
