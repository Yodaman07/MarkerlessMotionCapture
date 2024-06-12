import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from renderLandmark import renderLandmarks

# help from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#live-stream_1
model_path = "pose_landmarker_full.task"

PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    output_segmentation_masks=True
)

detector = vision.PoseLandmarker.create_from_options(options)
a = cv.imread("guy.jpg")
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=a)
result = detector.detect(img)

annotated = renderLandmarks(img.numpy_view(), result)
cv.imshow("a", cv.cvtColor(annotated, cv.COLOR_RGB2BGR))
cv.waitKey(0)
cv.destroyAllWindows()

# cam = cv.VideoCapture(0)
# if not cam.isOpened():
#     print("Unable to access camera")  # kill the program if the camera is not accessed
#     cam.release()
#     exit()
#
# while True:
#     retrieved, frame = cam.read()
#     if not retrieved:
#         print("Stream has likely ended")
#         break


        # mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # landmarker.detect_async(mp_img)

#     cv.imshow("stream", frame)
#
#     # https://stackoverflow.com/questions/5217519/what-does-opencvs-cvwaitkey-function-do <-- how waitKey works
#     if cv.waitKey(1) == ord("q"):  # gets the unicode value for q
#         break
#
# cam.release()
# cv.destroyAllWindows()
