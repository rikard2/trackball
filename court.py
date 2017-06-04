import numpy as np
import cv2
import random

colorized = cv2.imread('screen1.jpg')
tpl = cv2.imread('edge.png')
colorized = cv2.resize(colorized, (1022, 650))
res = cv2.matchTemplate(colorized,tpl,cv2.TM_CCOEFF)
print(res)
frame = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY);

# find the outer corners of the court
court_corners = [ 0, 0, 0, 0]

def process_court(i, contours):
    print('processing...')
    a = 0
    bottom_left = (9999999, -1)
    bottom_right = (-1, -1)
    top_left = (999999, 9999999)
    top_right = (-1, -1)

    for point in contours:
        x, y = ( point[0][0], point[0][1] )
        if x < bottom_left[0]:
            bottom_left = ( x, y )
        if x > bottom_right[0]:
            bottom_right = ( x, y )
        if y < top_left[1]:
            top_left = ( x, y )
        if y < top_right[1]:
            top_left = ( x, y )
        cv2.circle(colorized, (point[0][0], point[0][1]), 1, (0, 255, 0), -1)
        a = a + 1
    print("bottom left", bottom_left)
    cv2.circle(colorized, bottom_left, 10, (0, 0, 255), -1)
    cv2.circle(colorized, bottom_right, 10, (255, 0, 0), -1)
    cv2.circle(colorized, top_left, 10, (0, 255, 0), -1)


edged = cv2.Canny(frame, 10, 90)


contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
print("Contours found: ", len(contours))
print(frame.shape)

cv2.imshow('edged', edged)
i = 0
for c in sorted_contours:
    process_court(i, c)
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    #cv2.drawContours(colorized, [c], -1, color, 3)
    #print(c)
    i = i + 1
    cv2.imshow('Template', colorized)
    cv2.waitKey(0)
    if 0xFF == ord('q'):
        break
    

#while True:
#    cv2.imshow('Template', edged)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

cv2.destroyAllWindows()
