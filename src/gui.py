import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
import time

from wordfreq import top_n_list

from camera import Camera
from sarvam_tts import speak

COMMON_WORDS = [word.upper() for word in top_n_list("en", 50000)]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


LETTER_SPEECH = {
    "A": "Letter A",
    "B": "Letter B",
    "C": "Letter C",
    "D": "Letter D",
    "E": "Letter E",
    "F": "Letter F",
    "G": "Letter G",
    "H": "Letter H",
    "I": "Letter I",
    "J": "Letter J",
    "K": "Letter K",
    "L": "Letter L",
    "M": "Letter M",
    "N": "Letter N",
    "O": "Letter O",
    "P": "Letter P",
    "Q": "Letter Q",
    "R": "Letter R",
    "S": "Letter S",
    "T": "Letter T",
    "U": "Letter U",
    "V": "Letter V",
    "W": "Letter W",
    "X": "Letter X",
    "Y": "Letter Y",
    "Z": "Letter Z",

    "0": "Number Zero",
    "1": "Number One",
    "2": "Number Two",
    "3": "Number Three",
    "4": "Number Four",
    "5": "Number Five",
    "6": "Number Six",
    "7": "Number Seven",
    "8": "Number Eight",
    "9": "Number Nine"
}


class SignovaGUI:

    def __init__(self):

        self.root = ctk.CTk()

        self.root.title("✨ Signova")
        self.root.geometry("1600x900")
        self.root.configure(fg_color="#12131A")

        self.camera_feed = Camera()

        self.current_prediction = "-"
        self.current_confidence = 0

        self.last_spoken = ""
        self.last_spoken_time = 0
        self.speech_cooldown = 2.0

        # Prediction smoothing
        self.last_prediction = "-"
        self.prediction_count = 0
        self.stable_prediction = "-"
        self.required_frames = 12

        # Sentence builder
        self.sentence = ""

        self.last_added = ""
        self.last_added_time = 0
        self.auto_add_delay = 1.2   # seconds

        self.build_ui()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

    def build_ui(self):

        # ===========================
        # HEADER
        # ===========================

        header = ctk.CTkFrame(
            self.root,
            height=70,
            corner_radius=20,
            fg_color="#1D1F2B"
        )

        header.pack(fill="x", padx=15, pady=(15,8))

        logo = ctk.CTkLabel(
            header,
            text="🤟 Signova",
            font=("Poppins",30,"bold")
        )

        logo.pack(side="left", padx=25)

        subtitle = ctk.CTkLabel(
            header,
            text="AI Powered Sign Language Interpreter",
            font=("Poppins",15)
        )

        subtitle.pack(side="left", padx=15)

        theme = ctk.CTkButton(
            header,
            text="🌙",
            width=45,
            corner_radius=20
        )

        theme.pack(side="right", padx=15)

        settings = ctk.CTkButton(
            header,
            text="⚙",
            width=45,
            corner_radius=20
        )

        settings.pack(side="right")

        # ===========================
        # MAIN BODY
        # ===========================

        body = ctk.CTkFrame(
            self.root,
            fg_color="transparent"
        )

        body.pack(fill="both",expand=True,padx=15,pady=10)

        # -----------------------------
        # CAMERA PANEL
        # -----------------------------

        left = ctk.CTkFrame(
            body,
            fg_color="#1D1F2B",
            corner_radius=20
        )

        left.pack(side="left",fill="both",expand=True,padx=(0,10))

        camera_title = ctk.CTkLabel(
            left,
            text="📷 Live Camera",
            font=("Poppins",22,"bold")
        )

        camera_title.pack(pady=20)

        self.camera = ctk.CTkLabel(
            left,
            text="Camera Feed",
            width=950,
            height=650,
            fg_color="#2A2D3E",
            corner_radius=20,
            font=("Poppins",22)
        )

        self.camera.pack(pady=10)

        # -----------------------------
        # RIGHT PANEL
        # -----------------------------

        right = ctk.CTkFrame(
            body,
            width=420,
            fg_color="#1D1F2B",
            corner_radius=20
        )

        right.pack(side="right",fill="y")

        prediction_title = ctk.CTkLabel(
            right,
            text="Current Prediction",
            font=("Poppins",22,"bold")
        )

        prediction_title.pack(pady=(25,10))

        self.prediction = ctk.CTkLabel(
            right,
            text="-",
            font=("Poppins",72,"bold"),
            text_color="#7C5CFF"
        )

        self.prediction.pack()

        self.confidence = ctk.CTkProgressBar(
            right,
            width=300,
            progress_color="#6EEB83"
        )

        self.confidence.set(0)

        self.confidence.pack(pady=15)

        self.confidence_text = ctk.CTkLabel(
            right,
            text="Confidence : 0%",
            font=("Poppins",16)
        )

        self.confidence_text.pack()

        status_frame = ctk.CTkFrame(
            right,
            fg_color="#2A2D3E",
            corner_radius=15
        )

        status_frame.pack(fill="x",padx=20,pady=25)

        self.status = ctk.CTkLabel(
            status_frame,
            text="🟢 Ready",
            font=("Poppins",16,"bold")
        )

        self.status.pack(pady=10)

        sentence_title = ctk.CTkLabel(
            right,
            text="Sentence Builder",
            font=("Poppins",20,"bold")
        )

        sentence_title.pack()

        self.textbox = ctk.CTkTextbox(
            right,
            width=340,
            height=140,
            font=("Poppins",18),
            corner_radius=15
        )

        self.textbox.pack(pady=15)

        suggestions = ctk.CTkFrame(
            right,
            fg_color="transparent"
        )

        suggestions.pack()

        self.suggestion1 = ctk.CTkButton(
            suggestions,
            text="",
            width=100,
            state="disabled",
            command=lambda: self.use_suggestion(0)
        )
        self.suggestion1.grid(row=0,column=0,padx=5)

        self.suggestion2 = ctk.CTkButton(
            suggestions,
            text="",
            width=100,
            state="disabled",
            command=lambda: self.use_suggestion(1)
        )
        self.suggestion2.grid(row=0,column=1,padx=5)

        self.suggestion3 = ctk.CTkButton(
            suggestions,
            text="",
            width=100,
            state="disabled",
            command=lambda: self.use_suggestion(2)
        )
        self.suggestion3.grid(row=0,column=2,padx=5)

        controls = ctk.CTkFrame(
            right,
            fg_color="transparent"
        )

        controls.pack(pady=25)

        ctk.CTkButton(
            controls,
            text="➕ Add",
            width=150,
            height=45,
            fg_color="#7C5CFF",
            command=self.add_letter
        ).grid(row=0,column=0,padx=5,pady=5)

        ctk.CTkButton(
            controls,
            text="⎵ Space",
            width=150,
            height=45,
            fg_color="#3FB950",
            command=self.add_space
        ).grid(row=0,column=1,padx=5,pady=5)

        ctk.CTkButton(
            controls,
            text="⌫ Backspace",
            width=150,
            height=45,
            fg_color="#E3B341",
            command=self.backspace
        ).grid(row=1,column=0,padx=5,pady=5)

        ctk.CTkButton(
            controls,
            text="🗑 Clear",
            width=150,
            height=45,
            fg_color="#F85149",
            command=self.clear_text
        ).grid(row=1,column=1,padx=5,pady=5)

        self.speak = ctk.CTkButton(
            right,
            text="🔊 Speak",
            height=55,
            font=("Poppins",18,"bold"),
            fg_color="#9D4EDD",
            command=self.speak_sentence
        )

        self.speak.pack(fill="x",padx=30,pady=(0,25))

        self.update_suggestions()

    def update_camera(self):

        frame, prediction, confidence = self.camera_feed.get_frame()

        if frame is not None:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(rgb)

            image = image.resize((950,650))

            photo = ImageTk.PhotoImage(image)

            self.camera.configure(
                image=photo,
                text=""
            )

            self.camera.image = photo

            self.current_prediction = prediction
            self.current_confidence = confidence

            self.prediction.configure(
                text=prediction
            )

            # ---------- Prediction Smoothing ----------

            if prediction == self.last_prediction:
                self.prediction_count += 1
            else:
                self.last_prediction = prediction
                self.prediction_count = 1

            if self.prediction_count >= self.required_frames:
                self.stable_prediction = prediction

            display_text = self.stable_prediction

            if display_text == "-":
                display_text = "No Sign"

            self.prediction.configure(text=display_text)

            current_time = time.time()

            # --------------------------
            # AUTO ADD LETTER
            # --------------------------

            if (
                self.stable_prediction != "-"
                and self.stable_prediction != self.last_added
                and confidence > 0.90
            ):

                if current_time - self.last_added_time > self.auto_add_delay:

                    self.sentence += self.stable_prediction

                    self.textbox.delete("1.0", "end")
                    self.textbox.insert("1.0", self.sentence)

                    self.update_suggestions()

                    self.last_added = self.stable_prediction
                    self.last_added_time = current_time

                    threading.Thread(
                        target=speak,
                        args=(LETTER_SPEECH.get(self.stable_prediction, self.stable_prediction),),
                        daemon=True
                    ).start()

            self.confidence.set(confidence)

            self.confidence_text.configure(
                text=f"Confidence : {confidence*100:.1f}%"
            )

            if confidence > 0.80:

                self.status.configure(
                    text="🟢 Excellent Detection"
                )

            elif confidence > 0.60:

                self.status.configure(
                    text="🟡 Detecting..."
                )

            else:

                self.status.configure(
                    text="🔴 No confident prediction"

                )

        if prediction == "-":
            self.last_added = ""

        self.root.after(15, self.update_camera)

    # =====================================
    # SENTENCE BUILDER
    # =====================================

    def add_letter(self):

        if self.stable_prediction == "-":
            return

        self.sentence += self.stable_prediction

        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", self.sentence)

        self.update_suggestions()

    def add_space(self):

        self.sentence += " "

        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", self.sentence)

        self.update_suggestions()

    def backspace(self):

        self.sentence = self.sentence[:-1]

        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", self.sentence)

        self.update_suggestions()

    def clear_text(self):

        self.sentence = ""

        self.textbox.delete("1.0", "end")

        self.update_suggestions()

    def speak_sentence(self):

        text = self.textbox.get("1.0", "end").strip()

        if text == "":
            return

        threading.Thread(
            target=speak,
            args=(text,),
            daemon=True
        ).start()

    # =====================================
    # WORD SUGGESTIONS
    # =====================================

    def update_suggestions(self):

        current_word = self.sentence.split()[-1].upper() if self.sentence.strip() else ""

        matches = []

        if current_word:

            for word in COMMON_WORDS:

                if word.startswith(current_word):

                    matches.append(word)

                if len(matches) == 3:
                    break

        buttons = [
            self.suggestion1,
            self.suggestion2,
            self.suggestion3
        ]

        for i, button in enumerate(buttons):

            if i < len(matches):

                button.configure(
                    text=matches[i],
                    state="normal"
                )

            else:

                button.configure(
                    text="",
                    state="disabled"
                )

    def use_suggestion(self, index):

        buttons = [
            self.suggestion1,
            self.suggestion2,
            self.suggestion3
        ]

        word = buttons[index].cget("text")

        if word == "":
            return

        words = self.sentence.strip().split()

        if len(words) == 0:
            words = [word]
        else:
            words[-1] = word

        self.sentence = " ".join(words) + " "

        self.textbox.delete("1.0","end")
        self.textbox.insert("1.0", self.sentence)

        self.update_suggestions()

    def run(self):

        self.update_camera()

        self.root.mainloop()

    def on_close(self):

        self.camera_feed.close()

        self.root.destroy()