P2P Energy Trading Dashboard & Blockchain LedgerThis project is a complete Flask-based web application that simulates a Peer-to-Peer (P2P) energy trading platform. It features a secure user login, a live dashboard, real-time MQTT data integration, a persistent blockchain ledger for all trades, and automated execution of trading algorithms on user-uploaded datasets.Core FeaturesFull Authentication: Secure login and registration system (flask_login, bcrypt). The dashboard is a protected route.Live Dashboard: A dynamic frontend (using socketio) that shows live server stats, incoming MQTT trades, and blockchain status without needing to refresh.Trading Algorithm Execution: Users can upload a CSV dataset (e.g., HH001.csv) to trigger a Python-based trading algorithm (trading_algorithm.py) which runs in a background thread.Blockchain Ledger: All live trades (from MQTT) and algorithm results are logged to a "pending transaction" pool. A background task "mines" these transactions into new blocks every 20 seconds, saving them to a persistent ledger.json file.MQTT Integration: The server subscribes to iot-p2p-trading/trades on a public broker, logging all messages as potential transactions on the blockchain.Optimized API: The dashboard loads all initial data (entire blockchain, pending transactions, MQTT cache) in a single, efficient API call (/api/dashboard-data) to minimize server load.File StructureIOT Project Server/
├── app.py                      # Main Flask server, SocketIO, Login, API routes
├── blockchain_service.py       # Classes for Block and Blockchain (with PoW)
├── trading_algorithm.py      # Logic for the "Simple Auction" algorithm
├── ledger.json                 # (Auto-generated) Persistent blockchain data
├── requirements.txt            # All Python dependencies
├── templates/
│   ├── base.html               # Main site template (nav bar, styles)
│   ├── login.html              # Login & Register page
│   └── index.html              # Main dashboard UI
└── uploads/
    └── (Uploaded CSV files are stored here)
How It Works: System FlowUser Logs In: User visits http://127.0.0.1:5000, is redirected to /login, and either registers or logs in.Dashboard Loads: After login, index.html is rendered. Its JavaScript immediately makes one API call to /api/dashboard-data.Single API Call: app.py responds with a JSON object containing the entire blockchain, all pending transactions, and the last 10 MQTT messages.UI Populates: The JavaScript uses this data to fill in the "Live MQTT Feed" and "Immutable Trading Ledger" cards.Live Updates:SocketIO (update_data): The server's background thread pushes a new simulated "Trading Price" every 5 seconds.SocketIO (mqtt_message): When a new message arrives on the MQTT topic, app.py catches it, adds it to the blockchain's pending pool, and pushes it to all connected clients.SocketIO (blockchain_update): Every 20 seconds, the background thread "mines" all pending transactions (from MQTT and algorithm runs). If a new block is created, it pushes the block data to all clients, which triggers their dashboards to re-fetch the chain.User Runs Algorithm:User uploads HH001.csv via the form.app.py receives the file at /upload, saves it to the uploads/ folder, and starts a new background thread for the algorithm.trading_algorithm.py runs, processing all HH*.csv files in the uploads/ folder.When finished, app.py logs the summary to the blockchain's pending pool and pushes the result summary via the file_update SocketIO event.The dashboard displays the summary and its pending Transaction Hash (TX Hash).Installation & SetupClone/Download: Place all files in the IOT Project Server/ directory.Create Virtual Environment (Recommended):python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
Install Dependencies:Install all required Python packages from the requirements.txt file.pip install -r requirements.txt
Run the Server:python app.py
Access the App:Open your browser and go to http://127.0.0.1:5000.Login:Register a new user (e.g., user: test, pass: test).Or, use the default admin user:Username: adminPassword: adminHow to Use the DashboardLive Feeds: Watch the "Live Server Status" and "Live MQTT Feed" cards for real-time data.Run Algorithm: Upload one or more HH*.csv files to the "Run Trading Algorithm" card. The algorithm will run on all HH*.csv files in the uploads directory and display a summary.View Blockchain: The "Immutable Trading Ledger" shows all blocks. It updates automatically when a new block is mined (every 20 seconds) or when you press "Mine Now".Test MQTT: Use an MQTT client (like MQTT Explorer) to publish a JSON message to the iot-p2p-trading/trades topic on the mqtt.eclipse.org broker. It will appear instantly on the dashboard and be mined into the next block.Example Payload: {"buyer": "HH002", "seller": "HH001", "qty_kwh": 3.5, "price": 6.8}Developer: How to Add a Second Algorithm (for Comparison)The system is designed to be modular. You can easily add a second (or third) algorithm and run them side-by-side on the same dataset.Here is the step-by-step process, using trading_algorithm_2.py as an example.1. Create the New Algorithm FileCreate IOT Project Server/trading_algorithm_2.py. This file, just like the original, should be completely self-contained and have no knowledge of the blockchain or server. It just needs one main function that accepts a file path and returns a summary dictionary.Example: trading_algorithm_2.pyimport pandas as pd
import numpy as np
# ... other imports ...

def run_p2p_battery_simulation(uploaded_filepath):
    # ... your complex algorithm logic ...
    # (e.g., load data, simulate P2P with batteries)
    
    summary = {
        "files_processed": 10,
        "total_kwh_traded": 450.7,
        "total_community_profit": "₹850.20",
        "avg_p2p_price": "₹6.10",
        "total_unmet_demand_kwh": 25.3
    }
    
    print("Algo 2 (P2P Battery) finished.")
    return summary
2. Update app.py to Run BothModify app.py to import and call your new function.In app.py:# --- Import local modules ---
from blockchain_service import Blockchain
from trading_algorithm import run_trading_simulation
from trading_algorithm_2 import run_p2p_battery_simulation  # <-- 1. Import new function

# ...

def run_simulation_and_notify(filepath, username):
    """Wrapper to run BOTH simulations, emit results, and log to blockchain."""
    with app.app_context():
        summary_payload = {
            'filename': os.path.basename(filepath),
            'username': username,
            'status': 'processing'
        }
        try:
            # --- Run Algorithm 1: Simple Auction ---
            print("Running Algorithm 1: Simple Auction")
            summary1 = run_trading_simulation(filepath)
            
            # --- Run Algorithm 2: P2P with Battery ---
            print("Running Algorithm 2: P2P Battery")
            summary2 = run_p2p_battery_simulation(filepath) # <-- 2. Call new function

            # --- Combine results ---
            comparison_summary = {
                'simple_auction_v1': summary1,
                'p2p_battery_v1': summary2
            }
            
            summary_payload['status'] = 'processed'
            summary_payload['summary'] = comparison_summary # <-- 3. Send combined summary

            # --- Add the *combined* summary to the blockchain ---
            try:
                tx_data = {
                    'source': 'AlgorithmComparisonRun',
                    'user': username,
                    'input_file': os.path.basename(filepath),
                    'summary': comparison_summary # Log the full comparison
                }
                tx_hash = blockchain.add_transaction(tx_data)
                summary_payload['tx_hash'] = tx_hash
                print(f"Algorithm comparison logged to pending TXs with hash: {tx_hash}")
            except Exception as e:
                print(f"Error logging algorithm run to blockchain: {e}")
                summary_payload['tx_hash'] = None
            
            socketio.emit('file_update', summary_payload) # <-- 4. Emit to UI
            
        except Exception as e:
            # ... (error handling) ...
3. Update index.html to Display the ComparisonFinally, modify index.html to display the new comparison_summary object. You can create a comparison table, add charts, etc.In templates/index.html:You would modify the algo-results div to show a table instead of just <pre> text.The socket.on('file_update', ...) JavaScript function would be updated to parse data.summary.simple_auction_v1 and data.summary.p2p_battery_v1 and populate your new table.This design ensures your algorithm files (.py) remain clean and separate. All integration, encryption, and blockchain logic is handled by app.py (the "controller") and blockchain_service.py (the "model").API Endpoint DocumentationAll routes (except /login) require the user to be authenticated.GET /: Serves the main index.html dashboard.GET /login: Serves the login.html page.POST /login: Handles registration and login form submissions.GET /logout: Logs the user out and redirects to /login.POST /upload: Upload endpoint for the algorithm's CSV data.GET /api/dashboard-data: (Primary API) Returns a single JSON object with the full blockchain, pending transaction list, and MQTT log cache.POST /mine: Manually triggers the mining of a new block from pending transactions.GET /block/<index>: Returns the JSON data for a single, specific block from the blockchain.