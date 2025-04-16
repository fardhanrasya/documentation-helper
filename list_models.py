from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

# Konfigurasi API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Mendapatkan daftar model
models = genai.list_models()

# Menampilkan model yang tersedia
print("Daftar Model yang Tersedia:")
print("=" * 50)
for model in models:
    print(f"Nama Model: {model.name}")
    print(f"Versi: {model.version}")
    print(f"Deskripsi: {model.description}")
    print("-" * 50)
