import mediapipe as mp
import cv2 as cv

drawing_util = mp.solutions.drawing_utils
pose = mp.solutions.pose

cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Unable to access camera")  # kill the program if the camera is not accessed
    cam.release()
    exit()
with pose.Pose() as p:
    while True:
        retrieved, frame = cam.read()

        if not retrieved:
            print("Stream has likely ended")
            break

        result = p.process(frame)
        drawing_util.draw_landmarks(frame, result.pose_landmarks, pose.POSE_CONNECTIONS)

        cv.imshow("stream",frame)
        # https://stackoverflow.com/questions/5217519/what-does-opencvs-cvwaitkey-function-do <-- how waitKey works
        if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
            break

cam.release()
cv.destroyAllWindows()




