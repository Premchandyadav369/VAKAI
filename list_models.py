import google.generativeai as genai

API_KEY = "AIzaSyAEWFvpFp7xucyLmALx8U-L7QinaZ831Js"
genai.configure(api_key=API_KEY)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("FAILED:", e)
