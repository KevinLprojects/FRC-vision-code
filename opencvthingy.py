# import the opencv library 
import cv2
import numpy as np
from networktables import NetworkTables
from simple_pid import PID

NetworkTables.initialize(server="10.62.38.2")
sd = NetworkTables.getTable('vision')
sd.putNumber('right_joystick', 0)

p, i, d = .2, .05, .1
pid = PID(p, i, d, setpoint=.5)

cap = cv2.VideoCapture(0)
while True: 
    _, frame = cap.read()
    # It converts the BGR color space of image to HSV color space 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS) 
    

    lower_blue = np.array([3, 30, 130]) 
    upper_blue = np.array([12, 200, 255]) 
 
    # preparing the mask to overlay 
    mask = cv2.inRange(hsv, lower_blue, upper_blue)     

    threshhold_area = 5000
    window_size = 1
    contours, hierarchy = cv2.findContours(mask,  cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if contours and cv2.contourArea(contours[0]) > threshhold_area:
        contour = cv2.convexHull(contours[0])
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(frame, ellipse, (255,0,0), 3)
            control = pid(round((ellipse[0][0])/mask.shape[1], 2))
            #vision_publisher.set(np.tanh(control))
            sd.putNumber('right_joystick', np.tanh(control))
            #print((ellipse[0][0])/mask.shape[1])
            print('0', np.tanh(control))
    #     else:
    #         pid = PID(p, i, d, setpoint=.5)
    #         sd.putNumber('right_joystick', 0)
    #         print('1', 0)
    # else:
    #     pid = PID(p, i, d, setpoint=.5)
    #     sd.putNumber('right_joystick', 0)
    #     print('2', 0)

    
    #cv2.imshow('frame', mask)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(5)
    # waiting for q key to be pressed and then breaking
    if key == ord('q'):
        break
    