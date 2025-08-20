import os, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# paths
data_dir   = os.path.join("..", "data", "test")
model_try  = [os.path.join("..","models","emotion_model.keras"),
              os.path.join("..","models","emotion_model.h5")]

# load model (keras > h5 fallback)
model = None
for p in model_try:
    if os.path.exists(p):
        model = load_model(p)
        print(f"Loaded model: {p}")
        break
assert model is not None, "No saved model found in ../models/"

# test generator
datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    data_dir, target_size=(48,48), color_mode="grayscale",
    class_mode="categorical", shuffle=False, batch_size=64
)

# predictions
probs = model.predict(test_gen, verbose=0)
y_pred = np.argmax(probs, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# report
print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(os.path.join("..","models","confusion_matrix.png"))
print("Saved: ../models/confusion_matrix.png")
