import cv2
import mediapipe as mp
import math

def main():
    while True:
        success, img = cap.read()
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
