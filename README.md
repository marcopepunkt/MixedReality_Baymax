# MixedReality_Baymax
This Project Baymax is from the course Mixed Reality from ETHz in year 2024. A HoloLens 2 application is developed to assist viusally impaired users navigate in unknown environments.

This Repository contains code runing on the PC to receive and process data from Hololens. Code of the Unity app running on the Hololens is manitained in the other repository: https://github.com/marcopepunkt/MixedReality_Baymax_UnityApp

A demo video is avaiable on https://youtu.be/bi-2wldo6RU. More implementation and user study details of this project is avaiable in our report (link) and poster(link).

## Overview
By integrating real-time object detection, spatial audio and navigation features, our app aims to provide a safer and more independent experience in navigating unfamiliar environments with 3 functions:
1. Scene description
2. Obstacle avoidance
3. Navigation
Examples and an instruction for usage can be found in our demo video.

## Set-up 
1. Build our unity app from https://github.com/marcopepunkt/MixedReality_Baymax_UnityApp . We used Unity version 2020.3.42f1.
2. Clone this repository on your PC (recommended system: windows) and install dependencies from environment.yaml
3. Connect Hololens and your PC to the same network. set the IP address of the hololens in baymax.py
4. Set API keys (gemini, google maps) in baymax.py and run the script
5. after starting the unity app on hololens, say "Configure" to open a window where you insert the IP address of your PC, in the format "http::172.20.10.2:5000" (5000 is the port name of the hololens-PC connection)
6. to be able to use the google maps feature, follow the instructions on the unity app repository readme
7. now, you can make queries to baymax using the voice commands on welcome page (or on the unity app repository readme)
