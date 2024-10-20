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
using System.Diagnostics;
using System.Collections.ObjectModel;



// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace ServerApp
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private const int listenPort = 11000;
        private UdpClient listener;
        public ObservableCollection<string> LogMessages { get; set; }

        public MainPage()
        {
            this.InitializeComponent();
            LogMessages = new ObservableCollection<string>();
            LogListView.ItemsSource = LogMessages;
            StartListening();

        }

        private async void StartListening()
        {
            listener = new UdpClient(listenPort);
            IPEndPoint groupEP = new IPEndPoint(IPAddress.Any, listenPort);

            try
            {
                while (true)
                {
                    LogMessages.Add("Waiting for message");
                    UdpReceiveResult result = await listener.ReceiveAsync(); // Asynchronous receive
                    byte[] bytes = result.Buffer;

                    LogMessages.Add($"Received broadcast from {result.RemoteEndPoint} :");
                    LogMessages.Add($" {Encoding.ASCII.GetString(bytes, 0, bytes.Length)}");
                }
            }
            catch (SocketException e)
            {
                LogMessages.Add(e.ToString());
            }
            finally
            {
                listener.Close();
            }
        }
    }
}
