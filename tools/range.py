#!/usr/bin/env python
# -*- coding: utf-8 -*-

# USAGE: You need to specify a filter and "only one" image source
#
# (python) range-detector --filter RGB --image /path/to/image.png
# or
# (python) range-detector --filter HSV --webcam

import cv2
import math
import argparse
import numpy as np
from operator import xor

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Path to the image')
    ap.add_argument('-h1', '--fromh', required=True, help='From H')
    ap.add_argument('-s1', '--froms', required=True, help='From S')
    ap.add_argument('-v1', '--fromv', required=True, help='From V')
    ap.add_argument('-h2', '--toh', required=True, help='To H')
    ap.add_argument('-s2', '--tos', required=True, help='To S')
    ap.add_argument('-v2', '--tov', required=True, help='To V')
    args = vars(ap.parse_args())

    return args

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def main():
    args = get_arguments()

    #if args['image']:
    image = cv2.imread(args['image'])
    frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = (args['fromh'], args['froms'], args['fromv'], args['toh'], args['tos'], args['tov'])

    thresh = cv2.inRange(frame_to_thresh, (int(v1_min), int(v2_min), int(v3_min)), (int(v1_max), int(v2_max), int(v3_max)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(thresh,1,np.pi/180,100,minLineLength,maxLineGap)

    for x1,y1,x2,y2 in lines[0]:

        p1 = (x1, y1)
        p2 = (x2, y2)
        angle = angle_between(p1, p2)
        if angle > 1 and angle < 3:
            if abs(y2 - y1) < 20:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,255),2)
        print(angle, p1, p2)

    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 5:
            #cv2.drawContours(image, [c], -1, (0, 255, 255), 1)
            #print(peri, approx)
            pass

    #preview = cv2.bitwise_and(image, image, mask=thresh)
    #thresh = 

    cv2.imshow("Preview", cv2.resize(image, (810, 415)))
    #cv2.namedWindow('Preview',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Preview", 810, 415)
    while True:
        if cv2.waitKey(1) & 0xFF is ord('q'):
            break

if __name__ == '__main__':
    main()
