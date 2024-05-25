# %%
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# %%
gender_mapping = {
    1: 'Female',
    0: 'Male'
}

# %%
BASE_MODELS_DIR = "./use_models/"

faceProto = BASE_MODELS_DIR + 'opencv_face_detector.pbtxt'
faceModel = BASE_MODELS_DIR + 'opencv_face_detector_uint8.pb'
model_path = BASE_MODELS_DIR + 'model_58_3.1984989643096924.keras'

# %%
faceNet = cv2.dnn.readNet(faceModel, faceProto)
model = load_model(model_path)

# %%
def detectFace(net,frame,confidence_threshold=0.7):
    frameOpencvDNN=frame.copy()
    # print(frameOpencvDNN.shape)
    frameHeight=frameOpencvDNN.shape[0]
    frameWidth=frameOpencvDNN.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>confidence_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)
    return frameOpencvDNN,faceBoxes

# %%
video = cv2.VideoCapture(0)
padding = 20

import time

# Initialize variables for averaging age and gender
total_age = 0
total_gender = 0
count = 0
model_data = []

start_time = time.time()

face_frames = []
"""
[{
    "fame" : ,
    "facebox" : ,
}]
"""

len_to_predict = 16
result = None
need_to_cacl = False

while True:    
    # while time.time() - start_time < 10:
    hasFrame, raw_frame = video.read()

    if not hasFrame:
        cv2.waitKey()
        break
    
    if len(face_frames) <= len_to_predict:
        resultImg, faceBoxes = detectFace(faceNet, raw_frame)
        if faceBoxes:
            data = {
                "frame" : resultImg,
                "faceBoxes" : faceBoxes
            }
            face_frames.append(data)
            
        if len(face_frames) == 10:
            need_to_cacl = True
        
    if need_to_cacl: 
        for data in face_frames:
            # if not faceBoxes:
            #     continue

            for faceBox in data["faceBoxes"]:
                x = max(0, faceBox[1] - padding)
                w = min(faceBox[3] + padding, data["frame"].shape[0] - 1)
                y = max(0, faceBox[0] - padding)
                h = min(faceBox[2] + padding, data["frame"].shape[1] - 1)

                crop_face = data["frame"][x:w, y:h]

                crop_face = Image.fromarray(crop_face)
                crop_face = crop_face.resize((128, 128))
                crop_face = np.array(crop_face)
                crop_face = np.expand_dims(crop_face[:, :, 0], axis=-1)
                crop_face = crop_face.reshape(1, 128, 128, 1)
                crop_face = crop_face / 255.0

                pred = model.predict(crop_face)
                gender = gender_mapping[round(pred[0][0][0])]
                age = round(pred[1][0][0])

                total_age += age
                total_gender += 1 if gender == 'Female' else 0
                count += 1

        need_to_cacl = False
        result = (total_age // count, total_gender // count)
            # model_data.append(AgeAndGenderAverage)
    
    if result is not None:
        show_frame, faceBoxes = detectFace(faceNet, raw_frame)
        
        
        if len(faceBoxes) == 0 or not faceBoxes[0]:
            cv2.imshow("Model Data on Webcam", mat = raw_frame)
            continue
        
        cv2.putText(show_frame, f'{result[1]} -- {result[0]}',
                    (faceBoxes[0][0], faceBoxes[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Model Data on Webcam", show_frame)
    else:
        cv2.imshow("Model Data on Webcam", mat = raw_frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):
        face_frames = []
        result = None


"""
res = {
    "A" : [1, 50],
    "B" : [0, 30]
}

cv2.putText(resultImg, f'{res[1]}, {res[0]}',
                    (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Model Data on Webcam", resultImg)
"""