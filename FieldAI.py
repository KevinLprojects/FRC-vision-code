import cv2
import numpy as np
import time as time

cap = cv2.VideoCapture(0)

def find_match(image, subimage):
    #gray_subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)
    gray_subimage = subimage
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures = 500, fastThreshold = 5)
    kp1, des1 = orb.detectAndCompute(gray_subimage, None)
    orb = cv2.ORB_create(nfeatures = 500, fastThreshold = 5)
    kp2, des2 = orb.detectAndCompute(gray_image, None)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,
                    key_size = 12,
                    multi_probe_level = 1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    filtered_matches = []
    for i in matches:
        if len(i) == 2:
            filtered_matches.append(i)

    for i, m in enumerate(filtered_matches):
        m, n = m[0], m[1]
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    return cv2.drawMatchesKnn(subimage,kp1,image,kp2,matches,None,**draw_params)

subimage = cv2.imread('phone.jpg', 0) 
while True:
    _, image = cap.read()
    cv2.imshow('frame', find_match(image, subimage))
    if cv2.waitKey(1) == ord('q'):
        break