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


namespace ChatClient
{

	public class Client: Form
	{
		TcpClient newclient;
		NetworkStream clientstream;
		int start = 0;
		ASCIIEncoding encoder = new ASCIIEncoding ();
		byte[] buffer = new byte[4096];
		Button StartButton = new Button();
		TextBox sendBox = new TextBox();
		Button sendButton = new Button ();
		TextBox receiveBox = new TextBox();


		public Client()
		{
			this.SuspendLayout ();

			this.StartButton.Text = "Start";
			this.StartButton.Location = new System.Drawing.Point (4, 4);
			this.StartButton.Size = new System.Drawing.Size (60, 30);
			this.StartButton.Click += new System.EventHandler(this.startClient);

			this.sendBox.Location = new System.Drawing.Point (4, 50);
			this.sendBox.Size = new System.Drawing.Size (300, 30);
			this.sendBox.KeyDown += new KeyEventHandler (this.sendMsg);
			this.sendBox.Click += new EventHandler (this.gotFocused);



			this.sendButton.Location = new System.Drawing.Point (320, 50);
			this.sendButton.Size = new System.Drawing.Size (60, 30);
			this.sendButton.Text = "Send ";
			this.sendButton.Click += new System.EventHandler (this.sendData);

			this.receiveBox.Location = new System.Drawing.Point (4, 100);
			this.receiveBox.Size = new System.Drawing.Size (380, 260);
			this.receiveBox.Multiline = true;
			this.receiveBox.ScrollBars = ScrollBars.Vertical;
			this.receiveBox.ReadOnly = true;
			this.receiveBox.BackColor = Color.White;
	

		    

			this.Controls.Add (receiveBox);
			this.Controls.Add (sendButton);
			this.Controls.Add (StartButton);
			this.Controls.Add (sendBox);
			this.AutoScaleDimensions = new System.Drawing.SizeF (6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.Size = new System.Drawing.Size (450, 450);
			this.Text = "ChatClient";
			this.ResumeLayout ();


		}


		private void gotFocused(object sender, EventArgs e)
		{
			sendBox.Text = "";
		}

		private void sendMsg(object sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Enter) {
				clientstream = newclient.GetStream ();
				buffer = encoder.GetBytes (sendBox.Text);
				clientstream.Write (buffer, 0, buffer.Length);
				clientstream.Flush ();
				sendBox.Text = "";
			}
		}

		private void sendData(object sender, EventArgs e)
		{
			clientstream = newclient.GetStream ();


		    buffer = encoder.GetBytes (sendBox.Text);

			clientstream.Write (buffer, 0, buffer.Length);
			clientstream.Flush ();
			sendBox.Text = "";
		}

		private void startClient(object sender, EventArgs e)
		{
			if (start == 0) {
				start = 1;

				newclient = new TcpClient ("150.100.100.10", 8960);
				MessageBox.Show ("Connected");
				ClientReceive ();
			} 



		}

		public void ClientReceive()
		{
			clientstream = newclient.GetStream ();

			new Thread (() => 
			            {
				byte[] data = new byte[4096];
				int i;
				string strdata;

				while (true) {
					i = 0;
					i = clientstream.Read (data, 0, 4096);


					if (i == 0) {	break; }

					strdata = Encoding.ASCII.GetString (data, 0, i);
					if(receiveBox.Text == "")
					{
						receiveBox.Text = strdata;
					}
					else
					{
						receiveBox.Text = receiveBox.Text + Environment.NewLine + strdata;
					}
				}
			}).Start ();

		}
	}






	public class Program
	{
		public static void Main( string[] args)
		{
			Application.EnableVisualStyles ();
			Application.SetCompatibleTextRenderingDefault (false);
			Application.Run (new Client ());

		}

	}
}


