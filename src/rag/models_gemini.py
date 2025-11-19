import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

print("📋 MODELOS DISPONIBLES EN TU CUENTA:\n")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
        print(f"   Display: {model.display_name}")
        print(f"   Descripción: {model.description[:100]}...")
        print()