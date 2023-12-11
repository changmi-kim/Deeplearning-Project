import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from time import time

time1 = 0
THRESHOLD = 0.2 # 20%, 값이 클수록 손이 카메라와 가까워야 인식함
action_done = False

gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train_fy.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1) # 이미지 좌우 반전
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            if idx == 0 or idx == 6: # fist or six
                thumb_end = res.landmark[4]
                fist_end = res.landmark[17]

                text = None

                if thumb_end.x - fist_end.x > THRESHOLD:
                    text = 'RIGHT'
                    # pyautogui.moveTo(960,540)
                    pyautogui.keyUp('w')
                    pyautogui.moveTo(1160,540)
                    action_done = False

                elif fist_end.x - thumb_end.x > THRESHOLD:
                    text = 'LEFT'
                    # pyautogui.moveTo(960,540)
                    pyautogui.keyUp('w')
                    pyautogui.moveTo(760,540)
                    action_done = False

                elif thumb_end.y - fist_end.y > THRESHOLD:
                    text = 'DOWN'


                elif fist_end.y - thumb_end.y > THRESHOLD:
                    text = 'UP'
                    if action_done == False:
                        pyautogui.keyDown('w')
                        action_done = True

                if text is not None:
                    cv2.putText(img, text=text, org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 255, 0), thickness=10)
            elif idx in [0, 1, 2, 3, 4, 5, 6]:
                cv2.putText(img, text=str(idx), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 255, 0), thickness=10)
            
            if idx == 5:
                pyautogui.press('space')
                pyautogui.keyUp('w')
                action_done = False
            
            if idx == 11:
                x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

                fy_img = img[y1:y2, x1:x2].copy()
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
                fy_img = cv2.resize(fy_img, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                img[y1:y2, x1:x2] = fy_img
                
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    time2 = time()
    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)
        cv2.putText(img, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),3)
    time1 = time2

    cv2.imshow('Gesture\'s Law', img)
    if cv2.waitKey(1) == ord('q'):
        break
