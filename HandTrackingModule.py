import cv2
import mediapipe as mp
import math

width, height = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        param mode: In static mode, detection is done on each image: slower
        param maxHands: Maximum number of hands to detect
        param detectionCon: Minimum Detection Confidence Threshold
        param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils

        # Numbers correspond to tip of fingers
        # https://cs.opensource.google/mediapipe/mediapipe/+/master:mediapipe/python/solutions/hands.py
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []
        self.results = None

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.x * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0))
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

        if draw:
            return allHands, img
        else:
            return allHands



def main():

    detector = HandDetector(detectionCon=0.8, maxHands=2)
    success, img = cap.read()
    previous_hands, img = detector.findHands(img)

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        cv2.imshow("Image", img)

        if previous_hands and hands:

            hands1 = hands[0]
            hands2 = previous_hands[0]
            print("Hand")

            centerPoint1 = hands1['center']
            centerPoint2 = hands2['center']

            x1, y1 = centerPoint1
            x2, y2 = centerPoint2
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = math.hypot(x2 - x1, y2 - y1)

            cv2.putText(img, str(length), (250, 250), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

        previous_hands = hands
        key = cv2.waitKey(1)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        if key == ord('q'):
            break


main()
