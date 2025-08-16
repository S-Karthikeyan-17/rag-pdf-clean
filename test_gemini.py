import google.generativeai as genai

API_KEY = ""   # <--- paste here
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Hello Gemini! Can you confirm you're working?")

print(response.text)
