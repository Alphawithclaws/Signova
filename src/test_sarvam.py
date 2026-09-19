from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if api_key:
    print("✅ API Key loaded successfully!")
    print("First 10 characters:", api_key[:10] + "...")
else:
    print("❌ API Key not found.")