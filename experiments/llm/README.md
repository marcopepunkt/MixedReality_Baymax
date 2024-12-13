
### setup your own API key to run the code

1. Generate your own gemini API: https://aistudio.google.com/app/apikey
2. **Using a `.env` file in your project and paste your api key there**

```bash
*# Create a .env file, No quotes, no spaces around the equals sign. 
GOOGLE_API_KEY=your_actual_api_key_here*
```
3. Never commit API keys to version control! So add `.env` to your `.gitignore` file if you're using git


### Usage
1. after setting up the api, run the script to say hello and send a sample image to the genemi agent
```bash
python gemini_agent.py
```
or run following script to interact with the gemini agent with GUI
```bash
python gemini_gui.py
```

2. launch hl2ss app on hololens and then run
```bash
ython voice_triggered_description.py 
```
say 'describe' after the terminal shows 'Ready! Say 'describe' to get a scene description.', then you will hear a audio output from the PC reading the response from Gemini