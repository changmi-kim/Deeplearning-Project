from tensorflow import keras
from tensorflow.keras import models
import mediapipe as mp
import numpy as np
import cv2
import pyautogui 
import time

THRESHOLD = 0.2 # 20%, 값이 클수록 손이 카메라와 가까워야 인식함
action_ing = False
action_done = False

# 8가지 제스처 인식 Testv
# 영상용
# 한손만 인식
max_num_hands = 1
actions = ['go', 'back', 'stop', 'left_spin', 'right_spin', 'speed_up', 'speed_down', 'bad_gesture']
gestures = {
    0:'go', 1:'back', 2:'stop', 3:'left_spin', 4:'right_spin', 5:'speed_up',
    6:'speed_down', 7:'bad_gesture'}

# 시퀀스 길이 지정
seq_length = 4

# 미디어파이프 패키지에서 손 인식을 위한 객체 생성
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

model = keras.models.load_model("/home/ckdal/dev_ws/project/Dl_Project/model/LSTM_model_dataset100_epoch500.h5")

cap = cv2.VideoCapture(0)
seq = []
action_seq = []

fps = 0
frame_count = 0
start_time = time.time()

key_input_mode = 'OFF'
prev_action = '?'
current_action = '?'

action_ing = False
action_done = False

# increase/decrease only linear speed by 10%
speed = 0.2

while cap.isOpened():

    ret, img = cap.read()
    if not ret:
        print("카메라 연결 실패")
        # break
        continue

    height, width = img.shape[:2]
    border_thickness = 10
    border_thickness_all = 5

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # 1초마다 FPS 갱신
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    cv2.putText(img, 'Hand Gesture Detecting System', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(img, f'FPS: {int(fps)}', (510,460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Speed: {round(speed, 2)}', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 손 인식 여부 확인
    if result.multi_hand_landmarks is not None:

        for res in result.multi_hand_landmarks:
        
            joint = np.zeros((21, 4))
        
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1 # [20, 3]

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle)

            data = angle

            # data = np.array([angle], dtype=np.float32)

            seq.append(data)

            mp_drawing.draw_landmarks(img,
                                      res,
                                      mp_hands.HAND_CONNECTIONS
                                    #   mp_hands.get_default_hand_landmarks_style(),
                                    #   mp_hands.get_default_hand_connections_style()
            )
            
            if len(seq) < seq_length:
                continue

            # print(seq[-seq_length:])

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            # print(input_data)
            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 8:
                continue

            if len(action_seq) >= 8:
                prev_action = action_seq[-8]
                current_action = action_seq[-1]

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                this_num = i_pred
                action_done = False

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=7)
                    
            # key 입력
            if i_pred == 0:
                cv2.rectangle(img, (0, 0), (width, border_thickness), (0, 255, 0), -1)
                if action_done == False:
                    # pyautogui.keyDown('w')
                    action_ing = True
                    action_done = True
            
            elif i_pred == 1:
                cv2.rectangle(img, (0, height - border_thickness), (width, height), (0, 255, 255), -1)
                if action_done == False:
                    # pyautogui.keyDown('s')
                    action_ing = True
                    action_done = True

            elif i_pred == 2:
                cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), border_thickness_all)
                if action_ing == True:
                    # pyautogui.keyUp('w')
                    # pyautogui.keyUp('s')
                    action_ing = False

            elif i_pred == 3:
                cv2.rectangle(img, (0, 0), (border_thickness, height), (0, 127, 255), -1)
                if action_ing == True:
                    # pyautogui.keyUp('w')
                    # pyautogui.keyUp('s')
                    action_ing = False

                # pyautogui.moveTo(760, 1200)
                action_ing = False

            elif i_pred == 4:
                cv2.rectangle(img, (width - border_thickness, 0), (width, height), (0, 127, 255), -1)
                if action_ing == True:
                    # pyautogui.keyUp('w')
                    # pyautogui.keyUp('s')
                    action_ing = False

                # pyautogui.moveTo(1160, 1200)
                action_ing = False

            elif i_pred == 5:
                cv2.rectangle(img, (0, 0), (width, border_thickness), (0, 0, 255), -1)
                speed += speed * 0.1

            elif i_pred == 6:
                cv2.rectangle(img, (0, height - border_thickness), (width, height), (0, 0, 255), -1)
                speed -= speed * 0.1

            elif i_pred == 7:
                cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), border_thickness_all)
                
                if action_done == False:
                    # pyautogui.press('space')
                    action_ing = False
                    action_done = True
                
                x1, y1 = tuple((joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                x2, y2 = tuple((joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

                fy_img = img[y1:y2, x1:x2].copy()
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
                fy_img = cv2.resize(fy_img, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                img[y1:y2, x1:x2] = fy_img

        cv2.putText(img, f'Prev Action: {prev_action.upper()}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f'Current Action: {current_action.upper()}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Test', img)

    key = cv2.waitKey(5) & 0xFF

    if key == 27:
        cv2.destroyAllWindows()
        cap.release()
        break