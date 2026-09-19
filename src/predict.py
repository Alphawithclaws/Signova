import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
import threading
import time

from sarvam_tts import speak

# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = "model/sign_language.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# ============================================
# LOAD LABELS
# ============================================

DATASET_PATH = "dataset"

labels = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])

print("\nLoaded labels:")
print(labels)

print("\nModel output classes:", model.output_shape[-1])
print("Dataset classes:", len(labels))

if model.output_shape[-1] != len(labels):
    raise ValueError(
        f"\nMismatch detected!\n"
        f"Model expects {model.output_shape[-1]} classes\n"
        f"Dataset contains {len(labels)} folders\n"
        f"Retrain the model before predicting."
    )

# ============================================
# LETTERS TO SPEAK
# ============================================

LETTER_AUDIO = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "J",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "O",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "U": "U",
    "V": "V",
    "W": "W",
    "X": "X",
    "Y": "Y",
    "Z": "Z"
}

# ============================================
# MEDIAPIPE
# ============================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ============================================
# SPEECH SETTINGS
# ============================================

last_spoken = ""
last_time = 0
COOLDOWN = 2.0
CONFIDENCE_THRESHOLD = 0.90

# ============================================
# MAIN LOOP
# ============================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    label = "No hand"

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]

            x1 = max(int(min(xs) * w) - 30, 0)
            y1 = max(int(min(ys) * h) - 30, 0)

            x2 = min(int(max(xs) * w) + 30, w)
            y2 = min(int(max(ys) * h) + 30, h)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (224, 224))
            crop = crop.astype(np.float32) / 255.0
            crop = np.expand_dims(crop, axis=0)

            prediction = model.predict(crop, verbose=0)[0]

            index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            predicted_label = labels[index]

            label = f"{predicted_label} ({confidence*100:.1f}%)"

            current = time.time()

            if confidence >= CONFIDENCE_THRESHOLD:

                if (
                    predicted_label != last_spoken
                    or current - last_time > COOLDOWN
                ):

                    last_spoken = predicted_label
                    last_time = current

                    threading.Thread(
                        target=speak,
                        args=(LETTER_AUDIO.get(predicted_label, predicted_label),),
                        daemon=True
                    ).start()

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Alphabet Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()