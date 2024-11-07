
### setup your own API key to run the code

1. Generate your own gemini API: https://aistudio.google.com/app/apikey
2. **Using a `.env` file in your project and paste your api key there**

```bash
*# Create a .env file, No quotes, no spaces around the equals sign. 
GOOGLE_API_KEY=your_actual_api_key_here*
```
3. Never commit API keys to version control! So add `.env` to your `.gitignore` file if you're using git
4. run the script to say hello to the genemi agent
```bash
python helloworld.py
```