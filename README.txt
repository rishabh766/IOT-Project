P2P Energy Trading Data Server & Dashboard
This project provides the backend server and a minimalistic frontend dashboard for a Peer-to-Peer (P2P) energy trading platform. It's designed to accept user-submitted data files (like household energy consumption/generation), display real-time data, and provide a foundation for running complex trading algorithms.

The core infrastructure uses HTTP for serving the web interface and handling file uploads, WebSockets for pushing live data to all connected clients, and MQTT for subscribing to external data streams.

✨ Features
Flask Web Server: A robust Python backend to handle all requests.

File Upload Endpoint: Allows users to upload their data files (e.g., .csv) to the server via an HTTP POST request.

Real-time Dashboard: A WebSocket connection pushes live updates (like server time and mock trading prices) to the UI without needing to refresh the page.

MQTT Integration: The server subscribes to an MQTT topic to receive data from IoT devices or other services and broadcasts it to the dashboard.

Minimalistic UI: A clean, responsive frontend built with plain HTML, CSS, and JavaScript, featuring placeholders for future data visualizations.

Asynchronous Operations: Uses eventlet to efficiently manage concurrent WebSocket connections and the MQTT client loop.

🛠️ Tech Stack
Backend:

Python 3

Flask: Micro web framework.

Flask-SocketIO: WebSocket integration for Flask.

Paho-MQTT: MQTT client library.

Eventlet: Concurrency library for asynchronous I/O.

Frontend:

HTML5

CSS3

JavaScript (with Socket.IO Client)

📁 Project Structure
.
├── app.py              # Main Flask application, WebSocket server, and MQTT client
├── templates/
│   └── index.html      # Frontend UI file
└── uploads/
    └── (uploaded files will be stored here)
🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

Installation & Setup
Clone the repository (or download the files to a new directory):

Bash

git clone <your-repository-url>
cd <your-repository-folder>
Create and activate a virtual environment (recommended):

Bash

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
Install the required Python packages:

Bash

pip install "flask[async]" flask-socketio paho-mqtt eventlet
Running the Application
Start the server:

Bash

python app.py
Access the dashboard: Open your web browser and navigate to http://127.0.0.1:5000.

You should see the dashboard with live data updating every few seconds.

⚙️ How to Use
1. File Upload
On the dashboard, click the "Choose File" button in the "Upload Your Dataset" card.

Select a data file (e.g., HH001.csv).

Click the "Upload Data" button.

A status message will confirm if the upload was successful. The file will be saved in the uploads/ directory on the server.

2. Testing MQTT Integration
The server is subscribed to the trading/data topic on a public broker. You can publish a message to this topic to see it appear on the dashboard's "Live MQTT Feed".

You can use a command-line tool like mosquitto_pub or any other MQTT client.

Example using mosquitto_pub:

Bash

# Make sure you have mosquitto-clients installed
mosquitto_pub -h mqtt.eclipse.org -p 1883 -t "trading/data" -m '{"user": "user_A", "action": "buy", "quantity": 1.5}'
After running this command, the JSON message will instantly appear in the UI.

🏗️ Code Overview
Backend (app.py)
Initialization: Sets up Flask, SocketIO, and the uploads folder.

background_data_simulator(): A background thread that simulates real-time trading prices and the current time, emitting them via WebSockets every 5 seconds.

setup_mqtt_client(): Configures the Paho-MQTT client, defines on_connect and on_message callbacks, and connects to the broker. The on_message function is key, as it forwards messages from MQTT to all WebSocket clients.

HTTP Routes:

@app.route('/'): Serves the index.html file.

@app.route('/upload', methods=['POST']): Handles file uploads and saves them to the uploads/ directory.

SocketIO Events:

@socketio.on('connect'): Logs when a new client connects.

@socketio.on('disconnect'): Logs when a client disconnects.

Frontend (index.html)
Layout: The UI is structured using CSS for a simple card-based layout.

WebSocket Connection: The script connects to the server using io.connect().

Event Listeners:

socket.on('update_data', ...): Listens for the main data update event and updates the time and price placeholders.

socket.on('mqtt_message', ...): Listens for messages forwarded from the MQTT client and appends them to the log.

File Upload Logic: The form submission is handled by JavaScript's fetch API, which sends the file data to the /upload endpoint without a page reload.

