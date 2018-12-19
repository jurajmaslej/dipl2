import cv2
import numpy as np

filename = 'fog-data/MSG-d03-025-201206022200.jpg'

frame = cv2.imread(filename)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

cv::Scalar(157, 20, 20),cv::Scalar(184,255,255)
lower_pink = np.array([20, 20, 157])
upper_pink = np.array([184,255,255])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(frame,frame, mask= mask)

cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()
