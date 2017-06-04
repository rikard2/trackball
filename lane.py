import numpy as np
import cv2
import random

colorized = cv2.imread('screen1.jpg')
colorized = cv2.resize(colorized, (2044/2, 1100 /2))
frame = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY);

edged = cv2.Canny(frame, 10, 80)

#ret, thresh1 = cv2.threshold(frame, 127, 255, 0)
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
print("Contours found: ", len(contours))
print(frame.shape)

cv2.imshow('edged', edged)
for c in sorted_contours:
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.drawContours(colorized, [c], -1, color, 3)
    cv2.imshow('Template', colorizedd)
    cv2.waitKey(0)
    if 0xFF == ord('q'):
        break
    

#while True:
#    cv2.imshow('Template', edged)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

cv2.destroyAllWindows()
