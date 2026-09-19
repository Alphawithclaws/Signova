import cv2
import mediapipe as mp
import os

SIGN_NAME = "Z"

SAVE_PATH = os.path.join("dataset", SIGN_NAME)
os.makedirs(SAVE_PATH, exist_ok=True)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Couldn't open webcam.")
    exit()

count = len([
    f for f in os.listdir(SAVE_PATH)
    if f.endswith(".jpg")
])

print(f"\nSaving images to: {SAVE_PATH}")
print("Press S to save")
print("Press Q to quit\n")

while True:

    success, frame = cap.read()

    if not success:
        print("Camera read failed.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    hand_crop = None

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        h, w, _ = frame.shape

        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]

        x1 = max(int(min(xs) * w) - 40, 0)
        y1 = max(int(min(ys) * h) - 40, 0)

        x2 = min(int(max(xs) * w) + 40, w)
        y2 = min(int(max(ys) * h) + 40, h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        hand_crop = frame[y1:y2, x1:x2]

    cv2.putText(
        frame,
        f"Letter: {SIGN_NAME}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Images: {count}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Alphabet Dataset Collector", frame)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('s'):

        if hand_crop is not None and hand_crop.size > 0:

            image = cv2.resize(hand_crop, (224, 224))

            filename = os.path.join(
                SAVE_PATH,
                f"{count:04d}.jpg"
            )

            cv2.imwrite(filename, image)

            print("Saved:", filename)

            count += 1

        else:
            print("No hand detected!")

    elif key == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()