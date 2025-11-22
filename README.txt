IoT Peer-to-Peer (P2P) Energy Trading Platform
A robust, Flask-based web application designed to simulate a decentralized Peer-to-Peer energy trading market. This platform integrates Blockchain technology for immutable trade recording, MQTT for real-time IoT sensor data, and WebSockets for a live, reactive dashboard.

🚀 Features
Live Energy Dashboard: Real-time monitoring of server time, market status, and active nodes via WebSockets.

P2P Bidding Market: Users can publish energy offers and bid on available energy from neighbors in real-time.

Blockchain Ledger: A custom Python-based blockchain implementing Proof-of-Work (PoW) to securely log all trades and algorithm results.

Trading Algorithms: Capable of running complex Python simulations (e.g., Auction-based clearing) on uploaded household datasets (HH*.csv).

MQTT Integration: Subscribes to IoT topics (e.g., iot-p2p-trading/trades) to receive and display live sensor data.

User Authentication: Secure login and registration system storing user credentials in an Excel-based database (users.xlsx).

Dataset Generator: Includes tools to generate realistic synthetic load and solar generation data for Indian households using PVLib.

🛠️ Tech Stack
Backend: Python 3.8+, Flask, Flask-SocketIO, Flask-Login

Real-Time: WebSockets (Socket.IO), MQTT (Paho-MQTT)

Data & Storage:

Blockchain: Custom JSON-based ledger (ledger.json)

Database: Pandas/Excel (users.xlsx, trades.xlsx)

Frontend: HTML5, Tailwind CSS, JavaScript

Simulation: Pandas, NumPy, PVLib

📂 Project Structure
Plaintext

IOT-Project/
├── app.py                      # Main Application Server (Routes, Sockets, MQTT)
├── blockchain_service.py       # Blockchain Class (PoW, Hashing, Validation)
├── trading_algorithm.py        # Energy Trading Simulation Logic
├── simulated_sensors.py        # Script to mock IoT devices sending data
├── requirements.txt            # Python Dependencies
├── ledger.json                 # Persistent Blockchain Data
├── users.xlsx                  # User Credentials Database
├── uploads/                    # Directory for uploaded CSV datasets
├── templates/                  # HTML Templates (Jinja2)
│   ├── index.html              # Main Dashboard
│   ├── bidding.html            # Live Bidding Interface
│   └── login.html              # Auth Page
└── generate_dataset/           # Dataset Generation Tools
    └── generate_dataset.py     # Script to create synthetic household data
⚙️ Installation
Clone the repository:

Bash

git clone <repository-url>
cd IOT-Project
Create a Virtual Environment (Recommended):

Bash

python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
Install Dependencies:

Bash

pip install -r requirements.txt
🏃‍♂️ Running the Application
Start the Server:

Bash

python app.py
The server will start on http://0.0.0.0:5000 (accessible via localhost or network IP).

Access the Dashboard: Open your web browser and navigate to: http://127.0.0.1:5000

Login:

Default Admin: admin / admin (or register a new user).

🕹️ Usage Guide
1. Live Bidding
Navigate to the Live Bidding page.

Publish Offer: Enter Energy (kWh), Cost/Unit, and Min Bid Price to create a new market offer.

Place Bid: View active offers from other users and place bids before the 20-minute deadline expires.

Trade Execution: Winning bids are automatically executed, logged to Excel, and recorded on the Blockchain.

2. Running Simulations
Go to the Dashboard.

Upload a household dataset (e.g., HH001.csv from generate_dataset/example_dataset/).

The server will run the trading_algorithm.py in the background and display the results (Energy Traded, Cost Savings) via a live update.

3. Simulating IoT Sensors
To demonstrate MQTT integration without physical hardware:

Ensure app.py is running.

Open a new terminal.

Run the simulator script:

Bash

python simulated_sensors.py
Observe the Live MQTT Feed on the Dashboard updating with new "sensor" messages.

🔗 Configuration
MQTT Broker: Configured in app.py (Default: test.mosquitto.org, Port: 1883).

Blockchain Difficulty: Adjustable in app.py (Default: difficulty=2 for faster demonstration).

Dataset Generation: Modify parameters in generate_dataset.py to change location, household count, or prosumer ratios.

This project was developed for an IoT Energy Trading research initiative.