run from vision path
- `python .\real_time\get_depth_model.py` to get model_ov
- `python .\real_time\get_deptection_model.py` to get model
- `python .\receiver.py`

to run unity app and `baymax.py`:
- build and deploy unity app with usb/wifi, but run with wifi, for the connection from unity app to flask server. make sure hololens and pc are connected to the same network, and disable firewalls on windows (Windows Security > Firewall & network protection)
- update the ip address of the hololens on `baymax.py`.
- inside the unity app, write for the server url the ipv4 address of your pc (`ipconfig` on windows command prompt), and then ":5000/transform". you need to update the server url on `VoiceCommandHandler.cs`, `MyScript.cs` and also in `VoiceCommandHandler` and `WhatsAroundMe` objects in the unity Inspector window.
- update the path to the object detection model in `triggered_detection.py` if necessary
- now run `baymax.py` and start the unity app on hololens. don't wait too much in between; otherwise it can't initialize stream from hololens to pc.
- after the flask server is on, you can say "detect"

Azure TTS, Computer Vision, OpenAI:
- right now the code calls Azure TTS (in Unity app) and Azure Computer Vision resources
- make an account on Azure
- `Create a Resource` on left sidebar, search `Speech Services` (for TTS), `Computer Vision`, `OpenAI`. create one each, wait until they are deployed
- get their keys, resource names, write them on `triggered_detection.py` and `VoiceCommandHandler.cs` in unity app
