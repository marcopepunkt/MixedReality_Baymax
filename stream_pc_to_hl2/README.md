two simple apps (one for the server side and one for the client side) to stream a string of words from PC (client) to hololens (server).

i used the UDP protocol with the c# microsoft API functions, based on this example: https://learn.microsoft.com/en-us/dotnet/framework/network-programming/using-udp-services

instructions to run:

- clone/download both ClientApp and ServerApp
- find ClientApp/ClientApp/ClientApp.sln and ServerApp/ServerApp/ServerApp.sln and open in Visual Studio
- for ServerApp: this will run on the hololens. with usb: set ARM64, device on the top bar. build and deploy. (i already did this and the app is now on the hololens, not sure if we have to do it with every computer). 
- for ClientApp: this will run on the PC. set x64 (depends on your pc), local machine on the top bar. build, deploy.
- open the ServerApp on hololens and run the ClientApp on PC by debug > start without debugging
- on the Hololens GUI, it should appear first "waiting for message" and then "Hello HoloLens, this is a test message!"

other windows app samples for data streams (they implement both server and client in the same app though, so not sure how the stream would work):
- UDP: https://github.com/microsoft/Windows-universal-samples/tree/main/Samples/DatagramSocket
- TCP: https://github.com/Microsoft/Windows-universal-samples/tree/main/Samples/StreamSocket
- wifi direct: https://github.com/microsoft/windows-universal-samples/tree/main/Samples/WiFiDirect
- html: https://learn.microsoft.com/en-us/previous-versions/windows/apps/hh452986(v=win.10)?redirectedfrom=MSDN
