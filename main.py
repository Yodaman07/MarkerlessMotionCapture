import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# help from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#live-stream_1


model_path = "pose_landmarker_full.task"

PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
cam = cv.VideoCapture(0)


def callback(result: PoseLandmarkerResult, img: mp.Image, timestamp: int):
    print(result.pose_landmarks)
    cv.imshow("stream", img.numpy_view())


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=callback
)

time = 0
with vision.PoseLandmarker.create_from_options(
        options) as poseLandmarker:  # additional help from https://github.com/google-ai-edge/mediapipe/issues/
    while cam.isOpened():
        retrieved, frame = cam.read()
        if not retrieved:
            print("Stream has likely ended")
            break
        time += 1

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        poseLandmarker.detect_async(mp_img, time)
        # https://stackoverflow.com/questions/5217519/what-does-opencvs-cvwaitkey-function-do <-- how waitKey works
        if cv.waitKey(5) == ord("q"):  # gets the unicode value for q
            break

cam.release()
cv.destroyAllWindows()
