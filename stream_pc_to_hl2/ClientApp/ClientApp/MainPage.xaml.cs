using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.ServiceModel.Channels;
using System.Diagnostics;
using Windows.Networking.Sockets;
using Windows.Networking;
using Windows.Storage.Streams;
using System.Collections.ObjectModel;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace ClientApp
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public ObservableCollection<string> LogMessages { get; set; }
        public MainPage()
        {
            this.InitializeComponent();
            LogMessages = new ObservableCollection<string>();
            LogListView.ItemsSource = LogMessages;
            SendUdpMessage();
        }

        private async void SendUdpMessage()
        {
            // Create a new DatagramSocket
            DatagramSocket socket = new DatagramSocket();

            // Define the broadcast IP address and port
            string ip = "169.254.174.24"; // Replace with actual HoloLens IP or other target IP
            int port = 11000;

            try
            {
                // Connect to the remote IP and port
                HostName broadcast = new HostName(ip);
                string message = "Hello HoloLens, this is a test message!";
                byte[] sendbuf = Encoding.ASCII.GetBytes(message);

                // Get an output stream to send the message
                using (var stream = await socket.GetOutputStreamAsync(broadcast, port.ToString()))
                {
                    // Write the message to the output stream
                    using (DataWriter writer = new DataWriter(stream))
                    {
                        writer.WriteBytes(sendbuf);
                        await writer.StoreAsync();
                    }
                }

                LogMessages.Add("Message sent to the broadcast address");

            }
            catch (Exception ex)
            {
                LogMessages.Add("Error sending UDP message: " + ex.Message);
            }
            finally
            {
                socket.Dispose();  // Dispose of the socket after sending
            }
        }
    }
}
