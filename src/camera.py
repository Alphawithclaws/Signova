import cv2
import mediapipe as mp

from predictor import Predictor


class Camera:

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        self.predictor = Predictor()

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.drawer = mp.solutions.drawing_utils

    def get_frame(self):

        success, frame = self.cap.read()

        if not success:
            return None, "-", 0

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        prediction = "-"

        confidence = 0

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                self.drawer.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape

                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]

                x1 = max(int(min(xs) * w) - 30, 0)
                y1 = max(int(min(ys) * h) - 30, 0)

                x2 = min(int(max(xs) * w) + 30, w)
                y2 = min(int(max(ys) * h) + 30, h)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                crop = frame[y1:y2, x1:x2]

                prediction, confidence = self.predictor.predict(crop)

        return frame, prediction, confidence

    def close(self):

        self.cap.release()
        self.hands.close()