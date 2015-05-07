using System;
using System.IO;
using System.Timers;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace ChatServer
{
	class Server : Form
	{


		Button StartButton = new Button();
		TextBox ReceiveBox = new TextBox();
		TextBox SendBox = new TextBox();
		Button SendButton = new Button();
		private TcpListener listener = new TcpListener (IPAddress.Any,8960);
		private Thread listenThread;
		private int send = 0;
		TcpClient client;
		List<TcpClient> clientList = new List < TcpClient> () ;


		public Server ()
		{
			this.SuspendLayout ();

			this.StartButton.Text = "Start";
			this.StartButton.Location = new System.Drawing.Point (15, 4);
			this.StartButton.Size = new System.Drawing.Size (60, 30);
			this.StartButton.Click += new System.EventHandler(this.startServer);

			this.ReceiveBox.Location = new System.Drawing.Point (15, 100);
			this.ReceiveBox.Size = new System.Drawing.Size (400, 200);
			this.ReceiveBox.Multiline = true;
			this.ReceiveBox.ScrollBars = ScrollBars.Vertical;
			this.ReceiveBox.ReadOnly = true;
			this.ReceiveBox.BackColor = Color.White;

			this.SendBox.Location = new System.Drawing.Point (15, 50);
			this.SendBox.Size = new System.Drawing.Size (330, 30);
			this.SendBox.KeyDown += new KeyEventHandler (this.enterMsg);


			this.SendButton.Location = new System.Drawing.Point (350, 50);
			this.SendButton.Size = new System.Drawing.Size (60, 20);
			this.SendButton.Text = "Send";
			this.SendButton.Click += new System.EventHandler (this.sendMsg);


			this.Controls.Add (SendButton);
			this.Controls.Add (SendBox);
			this.Controls.Add (StartButton);
			this.Controls.Add (ReceiveBox);
			this.AutoScaleDimensions = new System.Drawing.SizeF (6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Size = new System.Drawing.Size (450, 450);
			this.ResumeLayout ();
			this.Text = "ChatServer";




		}



		private void startServer(object sender, EventArgs e)
		{

			listenThread = new Thread (new ThreadStart (ListenForClients));
			listenThread.Start ();
		}


		private void ListenForClients()
		{
			listener.Start ();
		    int counter = 0;

			while (true) 
			{
				client = listener.AcceptTcpClient ();
				counter++;
			
				clientList.Add (client);


				Thread clientThread = new Thread(delegate(){ handleClient(client,counter); });
				clientThread.Start ();


			}

		}

		private void sendMsg(object sender, EventArgs e)
		{
			send = 1;
		}

		private void enterMsg(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Enter) {
				send = 1;
			}
		}

		private void handleClient(object client,int i)
		{
			TcpClient chat = (TcpClient)client;
			NetworkStream chatStream = chat.GetStream ();

			byte[] message = new byte[4096];
			int byteRead;
			string data;

			ASCIIEncoding encode = new ASCIIEncoding ();
			byte[] sendData = new byte[4096];

			while (true) 
			{
				if (chatStream.DataAvailable == true) 
				{

					byteRead = 1;
					byteRead = chatStream.Read (message, 0, 4096);
					data = Encoding.ASCII.GetString (message, 0, byteRead);
					ReceiveBox.Text = data;


				}

				if (send == 1) {
					sendData = encode.GetBytes (SendBox.Text);
					chatStream.Write (sendData, 0, sendData.Length);
					chatStream.Flush ();
					SendBox.Text = "";
					send = 0;
				}




			}

			chat.Close ();
		}



	}



	public class Program
	{
		public static void Main(string[] args)
		{
			Application.EnableVisualStyles ();
			Application.SetCompatibleTextRenderingDefault (false);
			Application.Run (new Server ());

		}
	}
}
