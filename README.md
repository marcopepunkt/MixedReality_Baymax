run from vision path
- `python ..\real_time\get_depth_model.py` to get model_ov
- `python ..\real_time\get_deptection_model.py` to get model
- `python ..\receiver.py` #TODO: this scirpt does not exits anymore

# Detect
to run unity app and `baymax.py`:
1. **Configure Hololens IP in PC python script to receive data stream**: make sure hololens and pc are connected to the same network, and disable firewalls on windows (Windows Security > Firewall & network protection), update the ip address of the hololens on `baymax.py`. (usually 172.20.10.2, check in hololens settings)

2. **Configure PC IP in Unity IP to connect flask server**: inside the unity hub project, write for the server url the ipv4 address of your pc (`ipconfig` on windows command prompt), and then ":5000/transform". you need to update the server url on `VoiceCommandHandler.cs`, `MyScript.cs` and also in `VoiceCommandHandler` and `WhatsAroundMe` objects in the unity Inspector window.
3.  build and deploy unity app ´hl2ss unity´ with usb or wifi, but run the app with wifi, for the connection from unity app to flask server.
4. *Vision model choice*: (optional) update the path to the object detection model in `triggered_detection.py` if necessary
5. now run `baymax.py` and start the unity app on hololens. don't wait too much in between; otherwise it can't initialize stream from hololens to pc.
6. after the flask server is on, you can say "detect"

Azure TTS, Computer Vision, OpenAI:
- right now the code calls Azure TTS (in Unity app) and Azure Computer Vision resources
- make an account on Azure
- `Create a Resource` on left sidebar, search `Speech Services` (for TTS), `Computer Vision`, `OpenAI`. create one each, wait until they are deployed
- get their keys, resource names, write them on `triggered_detection.py` and `VoiceCommandHandler.cs` in unity app
## TODO
- add configure IP fct in app to avoid rebuild

# Describe
navigate to `./scripts`

## setup your own API key to run the code
1. Generate your own gemini API: https://aistudio.google.com/app/apikey
2. Using a .env file in your project and paste your api key there

```
# Create a .env file, No quotes, no spaces around the equals sign. 
GOOGLE_API_KEY=your_actual_api_key_here
```

3.Never commit API keys to version control! So add .env to your .gitignore file if you're using git

## Usage
1. after setting up the api, run the script to say hello and send a sample image to the genemi agent
`python gemini_agent.py`
or run following script to interact with the gemini agent with GUI
`python gemini_gui.py`
launch hl2ss app on hololens and then run
`python scene_descrption.py `

say 'describe' after the terminal shows 'Ready! Say 'describe' to get a scene description.', then you will hear a audio output from the PC reading the response from Gemini

