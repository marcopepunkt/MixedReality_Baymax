# MixedReality_Baymax
This Project Baymax is from the course Mixed Reality from ETHz in year 2024. A HoloLens 2 application is developed to assist viusally impaired users navigate in unknown environments.

This Repository contains code runing on the PC to receive and process data from Hololens. Code of the Unity app running on the Hololens is manitained in the other repository: https://github.com/marcopepunkt/MixedReality_Baymax_UnityApp

A demo video is avaiable on https://youtu.be/bi-2wldo6RU. More implementation and user study details of this project is available in our report and poster, which you can find in this repository.

## Overview
By integrating real-time object detection, spatial audio and navigation features, our app aims to provide a safer and more independent experience in navigating unfamiliar environments with 3 functions:
1. Scene description
2. Obstacle avoidance
3. Public transit directions

Examples and an instruction for usage can be found in our demo video.

## Instructions to build and run
1. Build our unity app from https://github.com/marcopepunkt/MixedReality_Baymax_UnityApp . We used Unity version 2020.3.42f1.
2. Clone this repository on your PC (recommended system: windows or MacOS) and install dependencies from `scripts/environment.yaml`
3. Connect Hololens and your PC to the same network. set the IP address of the hololens in `run.sh`. if you're using windows, you need to disable firewalls on your PC to be connected to the HoloLens over wifi.
4. Set API keys (gemini, google maps) in `run.sh`
5. to be able to use the google maps feature, follow the setup instructions in section Google maps below
6. start the unity app on hololens. say "Configure" to open a window where you insert the IP address of your PC, in the format "http://172.20.10.2:5000" (172.20.10.2 is the IP of the PC, 5000 is the port number of the hololens-PC connection in `scripts/baymax.py`. 5000 for windows OS, 6000 for MacOS)
7. run `run.sh` on your PC
8. after the PC server has started and the stream to hololens is initialized, you can make queries to baymax using the voice commands described below

## Voice commands
- "Configure" - To open UI to change IP 
- "Start" - Start Obstacle Avoidance
- "Stop" - End Obstacle Avoidance
- "Hey Baymax" - Activates scene description mode. After the ring, ask anything about the scene in front of you.
- "Abort" - Deactivates scene description mode.
- "Hey Baymax, take me to ... " - Activates google maps mode. Provide a location to get public transportation directions there.
- "Stop" / "Abort" - Deactivates google maps mode.

## Google Maps

### set up gps connection between phone and pc:
1. install GPS2IP Lite app on phone [GPS2IP](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://apps.apple.com/us/app/gps2ip-lite/id1562823492&ved=2ahUKEwjtvvLtoKKKAxXThv0HHX4zIIwQFnoECBcQAQ&usg=AOvVaw3MjoYW7jSYqW38cMqiVWUS).
2. inside the app: go to Settings > Connection Method, choose TCP Push
3. click on TCP Push and write the IP Address of your PC. The port number should be 11123 (should be the same on the app and on `receive_gps.py`)
4. in Settings > Network Selection, choose Cellular IP if you are connecting over your mobile data
5. on PC: download [Packet Sender](http://packetsender.com/)
6. inside the packet sender app; go to settings and enable TCP server. write down TCP server port (11123). instructions showing UI here under [Test that we can receive GPS2IP data](https://capsicumdreams.com/gps2ip/tcpPushMode.php). on the phone app, enable GPS2IP Lite on top of the main screen and follow the instructions from last url to check you are receiving packets on packet sender.
7. now you don't have to run packet sender app again when running baymax.py. just enable GPSIP Lite on the phone app each time you need gps coordinates.

### make requests to google maps while runnning the app:
1. say "hey baymax, take me to ..." -> provides instructions like tram line, time, departure stop
3. after hearing "Would you like additional instructions to first tram stop?", if you say "yes", it will give you the first walking instruction to the first tram stop and start receiving your gps coordinates from the phone. this feature is under testing/development and at the moment doesn't guide the user to final destination.

