import os
import json
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import paho.mqtt.client as mqtt
from threading import Thread
from collections import deque
import datetime
import random
import bcrypt
import eventlet
import pandas as pd
import openpyxl # Added for Excel file handling

# We need eventlet monkey patching for background tasks (MQTT + Data Sim)
eventlet.monkey_patch()

# --- Import local modules ---
from blockchain_service import Blockchain
from trading_algorithm import run_trading_simulation

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- START FIX: Use absolute path for user file ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# User Data File Configuration (New)
USER_FILE = os.path.join(BASE_DIR, 'users.xlsx')
# --- END FIX ---

# MQTT Broker settings
MQTT_BROKER = 'mqtt.eclipse.org'
MQTT_PORT = 1883
MQTT_TOPIC = 'iot-p2p-trading/trades'

# --- Flask & SocketIO App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'
socketio = SocketIO(app, async_mode='eventlet')

# --- Blockchain Initialization ---
blockchain = Blockchain(difficulty=2)

# --- App-level Cache ---
# Cache the last 10 MQTT messages to send on client load
MQTT_LOG_CACHE = deque(maxlen=10)


# =============================================================================
# --- User Authentication (Excel Persistence) ---
# =============================================================================

# --- Excel Utility Functions ---
def initialize_user_file():
    """Ensures the users.xlsx file exists with a header row and default admin."""
    # USER_FILE now uses the absolute path defined above
    if not os.path.exists(USER_FILE):
        try:
            # Generate a hashed password for the default admin
            hashed_admin_pass = bcrypt.hashpw(b'admin', bcrypt.gensalt()).decode('utf-8')
            
            # Create a DataFrame for the initial user data
            df = pd.DataFrame({
                'id': ['admin'],
                'password_hash': [hashed_admin_pass]
            })
            # Save the DataFrame to the Excel file
            df.to_excel(USER_FILE, index=False, engine='openpyxl')
            print(f"Initialized {USER_FILE} with default admin user.")
        except Exception as e:
            print(f"Error initializing user file: {e}")

def load_users_from_excel():
    """Loads all user data (ID and Hashed Password) from the Excel file into an in-memory dictionary."""
    try:
        # Read the Excel file. Assumes the first column is 'id' and second is 'password_hash'.
        df = pd.read_excel(USER_FILE, engine='openpyxl')
        
        # Convert the DataFrame into a dictionary for quick lookup: {'username': {'id': 'user', 'password_hash': '...'}}
        users_dict = {}
        for index, row in df.iterrows():
            user_id = str(row['id'])
            password_hash = str(row['password_hash'])
            users_dict[user_id] = {'id': user_id, 'password_hash': password_hash}

        return users_dict
        
    except FileNotFoundError:
        # If the file is missing, initialize it and try again
        initialize_user_file()
        return load_users_from_excel()
    except Exception as e:
        print(f"Error loading users from Excel: {e}. Returning empty user set.")
        return {} # Return empty on serious error

def save_new_user(user_id, password_hash):
    """Appends a new user to the Excel file."""
    try:
        # Load existing data, assuming it has been initialized correctly
        df_existing = pd.read_excel(USER_FILE, engine='openpyxl')
        
        # Prepare new user data
        new_user_df = pd.DataFrame({'id': [user_id], 'password_hash': [password_hash]})
        
        # Concatenate and save back to Excel, overwriting the old file
        df_updated = pd.concat([df_existing, new_user_df], ignore_index=True)
        df_updated.to_excel(USER_FILE, index=False, engine='openpyxl')
        
        # Update the in-memory cache
        USER_CACHE[user_id] = {'id': user_id, 'password_hash': password_hash}
        print(f"User {user_id} saved to {USER_FILE}.")
        return True
    except Exception as e:
        print(f"Error saving new user to Excel: {e}")
        return False
        
# --- Initialization ---
# This executes once on server startup
initialize_user_file() 
USER_CACHE = load_users_from_excel()
# --- End Excel Utility ---

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id
        # Fetch password hash from the in-memory cache
        self.password_hash = USER_CACHE.get(id, {}).get('password_hash')

    @staticmethod
    def get(user_id):
        # Check cache (which holds data loaded from Excel)
        if user_id in USER_CACHE:
            return User(user_id)
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


# --- Background Task for Real-time Data & Mining ---
# ... (rest of the file remains the same) ...
def background_data_simulator():
    """Simulates real-time data AND periodically mines the blockchain."""
    print("Starting background data simulator & miner...")
    mine_counter = 0
    while True:
        try:
            # 1. Simulate trading prices
            mock_price = round(100 + random.uniform(-5, 5), 2)
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data = {'time': current_time, 'price': mock_price}
            socketio.emit('update_data', data)

            # 2. Periodically mine a new block (e.g., every 4 data ticks = 20 seconds)
            if mine_counter % 4 == 0:
                if blockchain.pending_transactions:
                    print("Mining pending transactions...")
                    mined_block = blockchain.mine_pending_transactions()
                    if mined_block:
                        print(f"Block #{mined_block['index']} mined and saved to ledger.")
                        socketio.emit('blockchain_update', mined_block)

            mine_counter += 1
            socketio.sleep(5)  # Use socketio.sleep for eventlet
        except Exception as e:
            print(f"Error in background thread: {e}")
            break

def setup_mqtt_client():
    """Initializes and connects the MQTT client."""

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(MQTT_TOPIC)
            print(f"Subscribed to topic: {MQTT_TOPIC}")
        else:
            print(f"Failed to connect to MQTT, return code {rc}")

    def on_message(client, userdata, msg):
        """Callback for when a message is received from MQTT."""
        with app.app_context():  # Ensure we have app context for socketio
            try:
                payload_str = msg.payload.decode()
                print(f"MQTT: {msg.topic} -> {payload_str}")
                data = json.loads(payload_str)

                # Add to Blockchain's pending pool
                tx_hash = blockchain.add_transaction({
                    'source': 'MQTT',
                    'topic': msg.topic,
                    'data': data
                })

                # Create message payload
                message_payload = {'topic': msg.topic, 'payload': data, 'tx_hash': tx_hash}

                # Add to cache for new clients
                MQTT_LOG_CACHE.append(message_payload)

                # Broadcast live to all WebSocket clients
                socketio.emit('mqtt_message', message_payload)
            except Exception as e:
                print(f"Error in on_message: {e}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"Could not connect to MQTT broker: {e}")

    return client

# --- HTTP Routes (Updated Login Logic) ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        password = request.form.get('password')

        if action == 'register':
            # Check if user already exists using the USER_CACHE (loaded from Excel)
            if username in USER_CACHE:
                flash('Username already exists.', 'error')
            elif not username or not password:
                flash('Username and password are required.', 'error')
            else:
                hashed_pass = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                
                # Save the new user to the Excel file and update the cache
                if save_new_user(username, hashed_pass):
                    flash('Registration successful! Please log in.', 'success')
                    return redirect(url_for('login'))
                else:
                    flash('Registration failed due to a server error.', 'error')

        elif action == 'login':
            user = User.get(username)
            # Use the password hash retrieved from the USER_CACHE/Excel for verification
            if user and user.password_hash and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                login_user(user)
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """Serves the main protected dashboard (index.html)."""
    return render_template('index.html', username=current_user.id)


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handles file uploads and triggers the trading algorithm."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")

        try:
            # Run the algorithm in a new thread
            algo_thread = Thread(target=run_simulation_and_notify, args=(filepath, current_user.id))
            algo_thread.start()
        except Exception as e:
            print(f"Error starting algorithm thread: {e}")
            return jsonify({'error': f'Failed to start algorithm: {e}'}), 500

        return jsonify({'success': f'File {filename} uploaded. Processing... Check dashboard for updates.'}), 200


def run_simulation_and_notify(filepath, username):
    """Wrapper to run simulation, emit results, and log to blockchain."""
    with app.app_context():
        summary_payload = {
            'filename': os.path.basename(filepath),
            'username': username,
            'status': 'processing'
        }
        try:
            # --- Run Algorithm 1: Simple Auction ---
            print("Running Algorithm 1: Simple Auction")
            summary = run_trading_simulation(filepath)

            summary_payload['status'] = 'processed'
            summary_payload['summary'] = summary

            # --- Add the algorithm summary to the blockchain ---
            try:
                tx_data = {
                    'source': 'AlgorithmRun',
                    'user': username,
                    'algorithm': 'Simple Auction v1',
                    'input_file': os.path.basename(filepath),
                    'summary': summary
                }
                tx_hash = blockchain.add_transaction(tx_data)
                summary_payload['tx_hash'] = tx_hash  # Send tx_hash to the UI
                print(f"Algorithm run logged to pending transactions with hash: {tx_hash}")
            except Exception as e:
                print(f"Error logging algorithm run to blockchain: {e}")
                summary_payload['tx_hash'] = None
            # --- End of section ---

            socketio.emit('file_update', summary_payload)

        except Exception as e:
            print(f"Algorithm failed: {e}")
            summary_payload['status'] = 'error'
            summary_payload['error'] = str(e)
            socketio.emit('file_update', summary_payload)


# --- API Endpoints ---
# ... (API endpoints remain unchanged) ...
@app.route('/api/dashboard-data', methods=['GET'])
@login_required
def get_dashboard_data():
    """
    SINGLE API endpoint to load all initial data for the dashboard.
    This reduces server load by combining API calls.
    """
    chain_data = blockchain.get_chain_data()
    return jsonify({
        'blockchain': {
            'length': len(chain_data),
            'chain': chain_data,
            'pending_tx': len(blockchain.pending_transactions)
        },
        'mqtt_log': list(MQTT_LOG_CACHE)
    })


@app.route('/mine', methods=['POST'])
@login_required
def mine_block():
    """API endpoint to manually trigger mining."""
    print(f"Manual mine triggered by {current_user.id}")
    if not blockchain.pending_transactions:
        return jsonify({'message': 'No pending transactions to mine.'}), 400

    mined_block = blockchain.mine_pending_transactions()

    if mined_block:
        print(f"Block #{mined_block['index']} mined manually.")
        socketio.emit('blockchain_update', mined_block)
        return jsonify({'success': 'New block mined!', 'block': mined_block}), 200
    else:
        return jsonify({'error': 'Mining failed. See server logs.'}), 500


@app.route('/block/<int:index>', methods=['GET'])
@login_required
def get_block(index):
    """API endpoint to get data for a single block."""
    if index < 0 or index >= len(blockchain.chain):
        return jsonify({'error': 'Block index out of range.'}), 404

    block = blockchain.chain[index]
    return jsonify(block.to_dict()), 200


# --- WebSocket Events ---
# ... (WebSocket events remain unchanged) ...
@socketio.on('connect')
@login_required
def handle_connect():
    print(f"Client connected: {current_user.id} ({request.sid})")
    emit('welcome', {'message': f'Welcome, {current_user.id}!'}, room=request.sid)


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting server setup...")
    thread = Thread(target=background_data_simulator)
    thread.daemon = True
    thread.start()

    mqtt_client = setup_mqtt_client()

    print("Starting Flask-SocketIO server on http://127.0.0.1:5000")
    # Ensure debug=True is set for local development if desired, but use_reloader=False 
    # is required because of the background threads we are starting manually.
    socketio.run(app, host='127.0.0.1', port=5000, debug=True, use_reloader=False)