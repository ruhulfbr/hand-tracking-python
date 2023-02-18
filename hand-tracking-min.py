import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 #Previoues Time
cTime = 0 #Current time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, landMK in enumerate(handLms.landmark):
                # print(id, landMK)
                imgHeight, imgWidth, imgC = img.shape
                cx, cy = int(landMK.x * imgWidth), int(landMK.y * imgHeight)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                elif id==4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #Get Frame per secoend
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime


    #Display Text for Frame per secoend DOC link for openCv put text - https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
