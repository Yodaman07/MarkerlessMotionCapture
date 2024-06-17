import cv2 as cv
import numpy as np

# copied basic cam code from the main.py which was copied from one of my previous projects
# object detection code from https://www.youtube.com/watch?v=RaCwLrKuS1w and https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html

prevCircle = None
dist = lambda x1, y1, x2, y2: ((x1 - x2) ** 2) + ((y1 - y2) ** 2)  # measures the distance between 2 circles
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

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (17,17), 0)  # kernal explanation https://en.wikipedia.org/wiki/Kernel_(image_processing)
    # circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=75,
    #                           maxRadius=400)  # set min and max radius when ready to test with setup
    rows = blur.shape[0]
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.2, 500,
                              param1=100, param2=30,
                              minRadius=75, maxRadius=400)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for circle in circles[0, :]:
            if chosen is None: chosen = circle # initial index for chosen
            if prevCircle is not None:  # the 0th index is x and the 1st index is y
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(circle[0], circle[1], prevCircle[0],
                                                                                    prevCircle[1]):
                    chosen = circle
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)
        prevCircle = chosen

    cv.imshow("Stream", frame)

    if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
        break

cam.release()
cv.destroyAllWindows()
