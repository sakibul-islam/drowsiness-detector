from imutils import face_utils
import dlib
import cv2
from scipy.spatial import distance
import time
import params
from pygame import mixer


def get_max_area_rect(rects):
    if len(rects) == 0:
        return
    areas = []
    for rect in rects:
        areas.append(rect.area())
    return rects[areas.index(max(areas))]


def get_eye_aspect_ratio(eye):
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])
    return (vertical_1+vertical_2)/(horizontal*2)  # aspect ratio of eye


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

ls, le = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rs, re = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
ms, me = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

cap = cv2.VideoCapture(0)
drowsinessInitialized = False

mixer.init()
mixer.music.load("alert.wav")

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    # print(rects)
    time.sleep(0.1)

    for (i, rect) in enumerate(rects):
        # print(rect)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # print(shape)

        leftEye = shape[ls:le]
        rightEye = shape[rs:re]
        mouth = shape[ms:me]

        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 0), -1)

        for (x, y) in leftEye:
            cv2.circle(image, (x, y), 1, (255, 255, 0), -1)

        for (x, y) in rightEye:
            cv2.circle(image, (x, y), 1, (255, 255, 0), -1)

        for (x, y) in mouth:
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

        leftEyeRatio = get_eye_aspect_ratio(leftEye)
        rightEyeRatio = get_eye_aspect_ratio(rightEye)
        eyeRatioAvg = (leftEyeRatio + rightEyeRatio) / 2.0
        # print('leftEyeRatio:', leftEyeRatio, 'rightEyeRatio:', rightEyeRatio)
        print('eyeRatioAvg', eyeRatioAvg)

        if eyeRatioAvg < params.DROWSINESS_EYE_RATIO:
            # print('😴 drowsiness')
            if not drowsinessInitialized:
                drowsinessInitialized = True
                drowsinessInitializationTime = time.time()

            drowsinessDuration = time.time() - drowsinessInitializationTime
            print("drowsinessDuration: ", drowsinessDuration)
            if drowsinessDuration >= params.DROWSINESS_MIN_DURATION:
                print('😴😴😴 Sleepy')
                if not mixer.music.get_busy():
                    mixer.music.play()
        else:
            drowsinessInitialized = False
            mixer.music.stop()

    cv2.imshow("Monitor", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
