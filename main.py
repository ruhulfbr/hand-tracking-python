import cv2
import time

#Import Ditector module
from HandDitector import HandDitector as Ditector

#Start Codeing
cap = cv2.VideoCapture(0)
detectHands = Ditector()  #create ditector object
pTime = 0 #Previoues Time
cTime = 0 #Current time

while True:
    success, img = cap.read()
    img = detectHands.findHands(img)
    handPositions = detectHands.findPosition(img, draw=False)

    if len(handPositions) != 0:
        print(handPositions[4])


    #Get Frame per secoend
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    #Display Text for Frame per secoend DOC link for openCv put text - https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
