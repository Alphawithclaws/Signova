import os
import base64
import tempfile
import pygame

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

pygame.mixer.init()


def speak(text):

    response = client.text_to_speech.convert(
        text=text,
        target_language_code="en-IN",
        speaker="shubh",
        model="bulbul:v3"
    )

    # First audio returned by Sarvam
    audio_base64 = response.audios[0]

    audio_bytes = base64.b64decode(audio_base64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp.write(audio_bytes)
        temp_path = temp.name

    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.unload()
    os.remove(temp_path)