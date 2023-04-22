import cv2
import numpy as np
import os

path = 'examples'
orb = cv2.ORB_create(nfeatures=1000)

images = []
class_names = []
my_list = os.listdir(path)

for my_class in my_list:
    img_curr = cv2.imread(f'{path}/{my_class}', 0)
    images.append(img_curr)
    class_names.append(os.path.splitext(my_class)[0])


def find_descriptor(images):
    descriptor_list = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        descriptor_list.append(des)
    return descriptor_list


def find_id(img, descriptor_list, thres=15):
    kp2, des2 = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher()
    match_list = []
    final_value = -1
    try:
        for des in descriptor_list:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            match_list.append(len(good))
    except:
        pass

    if len(match_list) != 0:
        if max(match_list) > thres:
            final_value = match_list.index(max(match_list))
    return final_value


descriptor_list = find_descriptor(images)

cap = cv2.VideoCapture(0)

while True:
    succes, img2 = cap.read()
    img_original = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = find_id(img2, descriptor_list)
    if id != -1:
        cv2.putText(
            img_original, class_names[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('img2', img_original)
    cv2.waitKey(1)
0
