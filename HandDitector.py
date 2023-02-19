import cv2
import mediapipe as mp

class HandDitector:
    def __init__(self, mode=False, maxHands=2, complexity=1, detectConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectConfidence = detectConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectConfidence,
                                        self.trackConfidence)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmsList = []
        if self.results.multi_hand_landmarks:

            handLms = self.results.multi_hand_landmarks[handNo]
            for id, landMK in enumerate(handLms.landmark):
                imgHeight, imgWidth, imgChannel = img.shape
                cx, cy = int(landMK.x * imgWidth), int(landMK.y * imgHeight)
                # print(id, cx, cy)
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

                lmsList.append([id, cx, cy])

        return lmsList




