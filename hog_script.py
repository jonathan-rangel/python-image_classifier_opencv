import cv2 as cv
import numpy as np
import os

path = 'examples'

images = []
classNames = []
myList = os.listdir(path)
print('Total Classes Detected', len(myList))
for cl in myList:
    imgCur = cv.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findDes(images): 
    desList = []
    cell_size = (16, 16)  
    block_size = (2, 2)  
    nbins = 9
    for img in images:
        hog = cv.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
        hist = hog.compute(img)
        desList.append(hist)
    return desList

def findID(img, desList, thres = 15): 
    cell_size = (16, 16)  
    block_size = (2, 2)  
    nbins = 9
    hog = cv.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    hist = hog.compute(img)
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try: 
        for des in desList: 
            matches = bf.knnMatch(des, hist, k=2)
            good = []
            for m,n in matches: 
                if m.distance < 0.75 * n.distance: 
                    good.append([m])
            matchList.append(len(good))
    except: 
        pass
    
    if len(matchList) != 0 :
        if max(matchList) > thres : 
            finalVal = matchList.index(max(matchList))
    return finalVal

desList = findDes(images)
print(len(desList))

captura = cv.VideoCapture('Videos/Video 1.mp4')

while True:
    ret, imagen = captura.read()

    id = findID(imagen, desList)
    
    if id != -1:
        cv.putText(imagen, classNames[id], (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    if ret == True:
        cv.imshow('Video', imagen)
        #Presionar tecla ESC para salir
        if cv.waitKey(20) & 0xFF == 27:
            break
    else:
        break
captura.release()
cv.destroyAllWindows()