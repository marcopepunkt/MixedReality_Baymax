
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() 

gemini_api_key = os.getenv('GEMINI_API_KEY')
print(gemini_api_key)
#genai.configure(api_key=os.environ["API_KEY"])
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Hello!")
print(response.text)
