import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

IMG_SIZE = 64
BATCH_SIZE = 32

# ================= LOAD MODELS =================

eye_model = tf.keras.models.load_model("models/eye_state_cnn.h5")
mouth_model = tf.keras.models.load_model("models/mouth_yawn_cnn.h5")

# ================= DATA GENERATOR =================

datagen = ImageDataGenerator(rescale=1./255)

# ================= EYE DATASET =================

eye_test = datagen.flow_from_directory(
    "dataset/eye",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ================= MOUTH DATASET =================

mouth_test = datagen.flow_from_directory(
    "dataset/mouth",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ================= EYE MODEL EVALUATION =================

print("\n===== Eye State Detection Model =====")

loss, accuracy = eye_model.evaluate(eye_test)
print("Eye Model Accuracy:", accuracy * 100, "%")

pred_eye = eye_model.predict(eye_test)
pred_eye = (pred_eye > 0.5).astype(int)

print("\nConfusion Matrix (Eye Model)")
print(confusion_matrix(eye_test.classes, pred_eye))

print("\nClassification Report (Eye Model)")
print(classification_report(eye_test.classes, pred_eye))


# ================= MOUTH MODEL EVALUATION =================

print("\n===== Yawning Detection Model =====")

loss, accuracy = mouth_model.evaluate(mouth_test)
print("Mouth Model Accuracy:", accuracy * 100, "%")

pred_mouth = mouth_model.predict(mouth_test)
pred_mouth = (pred_mouth > 0.5).astype(int)

print("\nConfusion Matrix (Mouth Model)")
print(confusion_matrix(mouth_test.classes, pred_mouth))

print("\nClassification Report (Mouth Model)")
print(classification_report(mouth_test.classes, pred_mouth))