import cv2 as cv
import numpy as np


def BGR2HSV(color: []):
    col = np.uint8([[color]])  # pixel format
    return cv.cvtColor(col, cv.COLOR_BGR2HSV)


green = BGR2HSV([0, 255, 0])

print(green)
lower_bound = np.array([50, 50, 50])
upper_bound = np.array([80, 255, 255])

cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Unable to access the camera")
    cam.release()
    exit()

while True:
    retrieved, frame = cam.read()
    if not retrieved:
        print("An error occurred, the stream has likely ended")
        break

    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # ret, thresh = cv.threshold(frame_gray, 200, 255, 0)
    # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame_HSV, lower_bound, upper_bound) # b&w img
    coloredMask = cv.bitwise_and(frame_HSV, frame_HSV, mask=mask)
    ret, thresh = cv.threshold(mask, 200, 255, cv.THRESH_BINARY) #https://stackoverflow.com/questions/44378099/opencv-draw-contours-of-objects-in-the-binary-image
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv.imshow("Stream", frame)

    if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
        break

cam.release()
cv.destroyAllWindows()
