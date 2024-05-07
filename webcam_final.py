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

# %% [markdown]
# ### Load models

# %%
BASE_MODELS_DIR = "./use_models/"

faceProto = BASE_MODELS_DIR + 'opencv_face_detector.pbtxt'
faceModel = BASE_MODELS_DIR + 'opencv_face_detector_uint8.pb'
model_path = BASE_MODELS_DIR + 'model_58_3.1984989643096924.keras'

# %%
faceNet = cv2.dnn.readNet(faceModel, faceProto)
model = load_model(model_path)

# %% [markdown]
# ### Webcam

# %%
def detectFace(net,frame,confidence_threshold=0.7):
    frameOpencvDNN=frame.copy()
    print(frameOpencvDNN.shape)
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

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = detectFace(faceNet, frame)

    if not faceBoxes:
        continue

    for faceBox in faceBoxes:
        x = max(0,faceBox[1] - padding)
        w = min(faceBox[3] + padding, frame.shape[0] - 1)
        y = max(0,faceBox[0] - padding)
        h = min(faceBox[2] + padding, frame.shape[1]-1)

        crop_face = frame[x:w, y:h]
        # cv2.imshow('Face Detection', crop_face)

        crop_face = Image.fromarray(crop_face)
        crop_face = crop_face.resize((128, 128))
        crop_face = np.array(crop_face)
        crop_face = np.expand_dims(crop_face[:, :, 0], axis=-1)
        crop_face = crop_face.reshape(1, 128, 128, 1)
        crop_face = crop_face / 255.0

        pred = model.predict(crop_face)
        gender = gender_mapping[round(pred[0][0][0])]
        age = round(pred[1][0][0])

        cv2.putText(resultImg, f'{gender}, {age}',
                     (faceBox[0],faceBox[1]-10),
                     cv2.FONT_HERSHEY_SIMPLEX,
                     0.8,(0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and Gender", resultImg)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

# %%
cv2.destroyAllWindows()


