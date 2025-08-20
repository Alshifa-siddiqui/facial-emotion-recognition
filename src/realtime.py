import os, cv2, numpy as np
from tensorflow.keras.models import load_model

# load model (keras > h5 fallback)
model = None
for p in ["../models/emotion_model.keras","../models/emotion_model.h5"]:
    if os.path.exists(p):
        model = load_model(p)
        print(f"Loaded model: {p}")
        break
assert model is not None, "No saved model found."

# class names from your data folder (sorted so it matches generator order)
data_train = os.path.join("..","data","train")
EMOTIONS = sorted([d for d in os.listdir(data_train)
                   if os.path.isdir(os.path.join(data_train,d))])
print("Classes:", EMOTIONS)

# face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48)).astype("float32")/255.0
        roi = np.expand_dims(roi, axis=(0, -1))
        probs = model.predict(roi, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = f"{EMOTIONS[idx]} ({probs[idx]*100:.1f}%)"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.imshow("Facial Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
