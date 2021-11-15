import cv2
import keras
import tensorflow as tf
import numpy as np

def draw_border(img, pt1, pt2, arc_color, arc_thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), arc_color, arc_thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), arc_color, arc_thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, arc_color, arc_thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), arc_color, arc_thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), arc_color, arc_thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, arc_color, arc_thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), arc_color, arc_thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), arc_color, arc_thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, arc_color, arc_thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), arc_color, arc_thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), arc_color, arc_thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, arc_color, arc_thickness)


# need to learn about other cascade
frontFaceCascade = cv2.CascadeClassifier(r'E:\Emotion Detector\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')

kerasModel = tf.keras.models.load_model(r'E:\Emotion Detector\model\model-011.h5')

emotion_labels = ['angry','disgust','fear','happy','neutral', 'sad', 'surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #greyscaleimage
    grayScaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = frontFaceCascade.detectMultiScale(grayScaleFrame, scaleFactor=1.5, minNeighbors=3)
    for (x, y, w, h) in face_coordinates:
        
        # fetch each faces
        roi = grayScaleFrame[x:x+w, y:y+h]
        
        if roi.any():
            roi = cv2.resize(roi, (48,48))
            roi = tf.keras.preprocessing.image.img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            #analyze emotion
            prediction = kerasModel.predict(roi)
            label=emotion_labels[prediction.argmax()]
            
            if len(label) > 0:
                #draw rectangle around face
                # cv2.rectangle(frame, (x, y,), (x + w, y + h), (255, 255, 255), 1)

                #draw face arc
                draw_border(frame, (x, y,), (x + w, y + h), (255, 255, 255), 1, 5, 5)

                #draw text
                text = "Emotion =" + str(label) + "\n"

                fontScale = 0.4
                color = (255, 255, 255)
                thickness = 0
                font = cv2.FONT_HERSHEY_COMPLEX
                textWidth, textHeight = cv2.getTextSize(text, font, fontScale, thickness)
                line_height = textWidth[1] + 5

                for i, line in enumerate(text.split("\n")):
                    y0 = y + 10 + h + i * line_height
                    cv2.putText(frame, line, (x, y0), font,
                                fontScale, color, thickness, 2)

    # show image
    cv2.imshow('BGR', frame)

    # break condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
