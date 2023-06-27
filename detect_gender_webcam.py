from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
                    
# load model
model = load_model('model-pi.h5')

# open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['Laki-laki','Perempuan']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (150,150))
        face_crop = img_to_array(face_crop)
        face_crop = face_crop.astype("float") / 255.0
        face_crop = np.expand_dims(face_crop, axis=0)
        face_crop = np.vstack([face_crop])

        # apply gender detection on face
        result = model.predict(face_crop)
        Y = startY - 10 if startY - 10 > 10 else startY + 10 
        if result < 0.5:
            # write label and confidence above face rectangle
            cv2.putText(frame, classes[0], (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, classes[1], (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (240, 127, 228), 2)
            cv2.rectangle(frame, (startX,startY), (endX,endY), (240, 127, 228), 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
