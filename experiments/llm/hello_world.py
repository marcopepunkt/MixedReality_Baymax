
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() 
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)


model = genai.GenerativeModel("gemini-1.5-flash")


response = model.generate_content("Hello!")
print(response.text)

myfile = genai.upload_file("./images/tram.jpg")
result = model.generate_content(
    [myfile, "\n\n", "Describe this photo to a blind person to help them understand where they are and for navigation purposes."],
)
print(result.text)