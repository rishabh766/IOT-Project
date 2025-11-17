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

# We need eventlet monkey patching for background tasks (MQTT + Data Sim)
eventlet.monkey_patch()

# --- Import local modules ---
from blockchain_service import Blockchain
from trading_algorithm import run_trading_simulation

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# --- User Authentication (Flask-Login) ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory user store (replace with a database for production)
hashed_admin_pass = bcrypt.hashpw(b'admin', bcrypt.gensalt()).decode('utf-8')
USERS = {
    "admin": {
        "id": "admin",
        "password_hash": hashed_admin_pass
    }
}


class User(UserMixin):
    def __init__(self, id):
        self.id = id
        self.password_hash = USERS.get(id, {}).get('password_hash')

    @staticmethod
    def get(user_id):
        if user_id in USERS:
            return User(user_id)
        return None


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


# --- Background Task for Real-time Data & Mining ---
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


# --- MQTT Client Setup ---
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


# --- HTTP Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        password = request.form.get('password')

        if action == 'register':
            if username in USERS:
                flash('Username already exists.', 'error')
            elif not username or not password:
                flash('Username and password are required.', 'error')
            else:
                hashed_pass = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                USERS[username] = {"id": username, "password_hash": hashed_pass}
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))

        elif action == 'login':
            user = User.get(username)
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
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
    socketio.run(app, host='127.0.0.1', port=5000, debug=True, use_reloader=False)