import os
import json
import uuid
import pandas as pd
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import paho.mqtt.client as mqtt
from threading import Thread, Lock
from collections import deque
import datetime
from datetime import timedelta
import random
import bcrypt
import eventlet
import openpyxl

# Eventlet monkey patching for concurrent background tasks
eventlet.monkey_patch()

# --- Import local modules ---
from blockchain_service import Blockchain
from trading_algorithm import run_trading_simulation

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
USER_FILE = os.path.join(BASE_DIR, 'users.xlsx')
TRADES_FILE = os.path.join(BASE_DIR, 'trades.xlsx')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MQTT Broker settings
MQTT_BROKER = 'test.mosquitto.org'
MQTT_PORT = 1883
MQTT_TOPIC = 'iot-p2p-trading/trades'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super-secret-key-change-in-prod'
socketio = SocketIO(app, async_mode='eventlet')

blockchain = Blockchain(difficulty=2)
MQTT_LOG_CACHE = deque(maxlen=10)

# --- In-Memory Bid Storage ---
ACTIVE_BIDS = {}
BIDS_LOCK = Lock()

# =============================================================================
# --- User Authentication & Excel Logic ---
# =============================================================================

USER_CACHE = {}

def initialize_user_file():
    """Ensures the users.xlsx file exists with a default admin user."""
    if not os.path.exists(USER_FILE):
        try:
            hashed_admin_pass = bcrypt.hashpw(b'admin', bcrypt.gensalt()).decode('utf-8')
            df = pd.DataFrame({
                'id': ['admin'],
                'password_hash': [hashed_admin_pass]
            })
            df.to_excel(USER_FILE, index=False, engine='openpyxl')
            print(f"Initialized {USER_FILE} with default admin user.")
        except Exception as e:
            print(f"Error initializing user file: {e}")

def load_users_from_excel():
    """Loads user data from Excel into memory."""
    try:
        df = pd.read_excel(USER_FILE, engine='openpyxl')
        users_dict = {}
        for index, row in df.iterrows():
            user_id = str(row['id'])
            password_hash = str(row['password_hash'])
            users_dict[user_id] = {'id': user_id, 'password_hash': password_hash}
        return users_dict
    except FileNotFoundError:
        initialize_user_file()
        return load_users_from_excel()
    except Exception as e:
        print(f"Error loading users: {e}")
        return {}

def save_new_user(user_id, password_hash):
    """Appends a new user to the Excel file and updates cache."""
    try:
        # Load existing
        if os.path.exists(USER_FILE):
            df_existing = pd.read_excel(USER_FILE, engine='openpyxl')
        else:
            df_existing = pd.DataFrame(columns=['id', 'password_hash'])
        
        # Add new
        new_user_df = pd.DataFrame({'id': [user_id], 'password_hash': [password_hash]})
        df_updated = pd.concat([df_existing, new_user_df], ignore_index=True)
        
        # Save
        df_updated.to_excel(USER_FILE, index=False, engine='openpyxl')
        
        # Update Cache
        USER_CACHE[user_id] = {'id': user_id, 'password_hash': password_hash}
        print(f"User {user_id} saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving new user: {e}")
        return False

def log_trade_to_excel(trade_data):
    """Logs a finalized trade to trades.xlsx."""
    try:
        if not os.path.exists(TRADES_FILE):
            df = pd.DataFrame(columns=['Timestamp', 'Bid_ID', 'Seller', 'Buyer', 'Energy_kWh', 'Rate_per_kWh', 'Total_Cost', 'TX_Hash'])
            df.to_excel(TRADES_FILE, index=False, engine='openpyxl')
        
        df_existing = pd.read_excel(TRADES_FILE, engine='openpyxl')
        new_row = pd.DataFrame([trade_data])
        df_updated = pd.concat([df_existing, new_row], ignore_index=True)
        df_updated.to_excel(TRADES_FILE, index=False, engine='openpyxl')
        print(f"Trade {trade_data['Bid_ID']} logged to Excel.")
    except Exception as e:
        print(f"Error logging trade to Excel: {e}")

# --- Initialize Users on Startup ---
initialize_user_file()
USER_CACHE = load_users_from_excel()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id
        self.password_hash = USER_CACHE.get(id, {}).get('password_hash')

    @staticmethod
    def get(user_id):
        if user_id in USER_CACHE:
            return User(user_id)
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# =============================================================================
# --- Background Tasks ---
# =============================================================================

def background_bid_monitor():
    """Checks for expired bids every second and executes trades."""
    print("Starting Bid Monitor...")
    while True:
        with BIDS_LOCK:
            now = datetime.datetime.now()
            expired_ids = []
            
            for bid_id, offer in ACTIVE_BIDS.items():
                if offer['status'] == 'active' and offer['deadline'] <= now:
                    expired_ids.append(bid_id)
                    
            for bid_id in expired_ids:
                offer = ACTIVE_BIDS[bid_id]
                print(f"Bid {bid_id} expired.")
                
                if offer['highest_bidder']:
                    print(f"Executing Trade: {offer['seller']} -> {offer['highest_bidder']}")
                    
                    tx_data = {
                        'source': 'P2P_Market_Finalize',
                        'type': 'TRADE',
                        'bid_id': bid_id,
                        'seller': offer['seller'],
                        'buyer': offer['highest_bidder'],
                        'energy': offer['energy'],
                        'price_per_unit': offer['highest_bid'],
                        'total_cost': float(offer['energy']) * float(offer['highest_bid']),
                        'timestamp': now.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # 1. Blockchain
                    tx_hash = blockchain.add_transaction(tx_data)
                    
                    # 2. Excel
                    excel_data = {
                        'Timestamp': tx_data['timestamp'],
                        'Bid_ID': bid_id,
                        'Seller': tx_data['seller'],
                        'Buyer': tx_data['buyer'],
                        'Energy_kWh': tx_data['energy'],
                        'Rate_per_kWh': tx_data['price_per_unit'],
                        'Total_Cost': tx_data['total_cost'],
                        'TX_Hash': tx_hash
                    }
                    log_trade_to_excel(excel_data)
                    
                    # 3. Notify
                    socketio.emit('bid_finalized', {
                        'bid_id': bid_id,
                        'winner': offer['highest_bidder'],
                        'amount': tx_data['total_cost'],
                        'tx_hash': tx_hash
                    })
                else:
                    socketio.emit('bid_expired', {'bid_id': bid_id})

                del ACTIVE_BIDS[bid_id]
                
        socketio.sleep(1)

def background_data_simulator():
    """Simulates server time and tracks the market top bidder."""
    while True:
        try:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            top_bidder_info = "No Active Bids"

            # Calculate the Highest Bidder across ALL active offers
            with BIDS_LOCK:
                if ACTIVE_BIDS:
                    # Filter offers that have at least one bid
                    active_with_bids = [
                        offer for offer in ACTIVE_BIDS.values() 
                        if offer['highest_bidder'] and offer['status'] == 'active'
                    ]
                    
                    if active_with_bids:
                        # Find the offer with the absolute highest bid amount
                        top_offer = max(active_with_bids, key=lambda x: x['highest_bid'])
                        top_bidder_info = f"{top_offer['highest_bidder']} (Bid #{top_offer['id']})"
                    elif len(ACTIVE_BIDS) > 0:
                        top_bidder_info = "No Bids Yet"

            # Emit the new data structure
            socketio.emit('update_data', {'time': current_time, 'top_bidder': top_bidder_info})
            
            socketio.sleep(5)
        except Exception as e:
            print(f"Error in simulator: {e}")
            break

def setup_mqtt_client():
    client = mqtt.Client()
    def on_message(client, userdata, msg):
        with app.app_context():
            try:
                data = json.loads(msg.payload.decode())
                tx_hash = blockchain.add_transaction({'source': 'MQTT', 'data': data})
                payload = {'topic': msg.topic, 'payload': data, 'tx_hash': tx_hash}
                MQTT_LOG_CACHE.append(payload)
                socketio.emit('mqtt_message', payload)
            except: pass
            
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except: pass
    return client

# =============================================================================
# --- Routes ---
# =============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    global USER_CACHE  # Ensure we use the global variable
    
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        password = request.form.get('password')

        if action == 'register':
            if username in USER_CACHE:
                flash('Username already exists.', 'error')
            elif not username or not password:
                flash('Username and password required.', 'error')
            else:
                hashed_pass = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                if save_new_user(username, hashed_pass):
                    flash('Registration successful! Please log in.', 'success')
                    return redirect(url_for('login'))
                else:
                    flash('Error saving user.', 'error')

        elif action == 'login':
            user = User.get(username)
            if user and user.password_hash and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                login_user(user)
                return redirect(url_for('index'))
            else:
                flash('Invalid credentials.', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=current_user.id)

@app.route('/bidding')
@login_required
def bidding():
    display_bids = []
    with BIDS_LOCK:
        for bid_id, offer in ACTIVE_BIDS.items():
            display_bids.append({
                'id': bid_id,
                'seller': offer['seller'],
                'energy': offer['energy'],
                'cost_per_kwh': offer['cost_per_kwh'],
                'min_price': offer['min_price'],
                'current_highest': offer['highest_bid'] if offer['highest_bid'] > 0 else offer['min_price'],
                'highest_bidder': offer['highest_bidder'],
                'deadline': offer['deadline'].isoformat()
            })
    return render_template('bidding.html', offers=display_bids, username=current_user.id)

@app.route('/api/create_offer', methods=['POST'])
@login_required
def create_offer():
    data = request.json
    try:
        energy = float(data.get('energy'))
        cost_per_kwh = float(data.get('cost_per_kwh'))
        min_price = float(data.get('min_price'))
        
        bid_id = str(uuid.uuid4())[:8]
        deadline = datetime.datetime.now() + timedelta(minutes=20)
        
        new_offer = {
            'id': bid_id,
            'seller': current_user.id,
            'energy': energy,
            'cost_per_kwh': cost_per_kwh,
            'min_price': min_price,
            'highest_bid': 0.0,
            'highest_bidder': None,
            'deadline': deadline,
            'status': 'active'
        }
        
        with BIDS_LOCK:
            ACTIVE_BIDS[bid_id] = new_offer
            
        payload = new_offer.copy()
        payload['deadline'] = deadline.isoformat()
        socketio.emit('new_offer', payload)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/place_bid', methods=['POST'])
@login_required
def place_bid_api():
    data = request.json
    offer_id = data.get('offer_id')
    bid_amount = float(data.get('bid_price'))

    with BIDS_LOCK:
        if offer_id not in ACTIVE_BIDS:
            return jsonify({'error': 'Offer not found'}), 404
        
        offer = ACTIVE_BIDS[offer_id]
        
        if offer['seller'] == current_user.id:
            return jsonify({'error': 'Cannot bid on own offer'}), 403
            
        current_threshold = max(offer['min_price'], offer['highest_bid'])
        if bid_amount <= current_threshold:
            return jsonify({'error': f'Bid must be > {current_threshold}'}), 400
            
        offer['highest_bid'] = bid_amount
        offer['highest_bidder'] = current_user.id
        
        socketio.emit('bid_update', {'bid_id': offer_id, 'new_price': bid_amount, 'bidder': current_user.id})

    return jsonify({'success': True}), 200

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No filename'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    Thread(target=run_simulation_and_notify, args=(filepath, current_user.id)).start()
    return jsonify({'success': f'Processing {file.filename}'}), 200

def run_simulation_and_notify(filepath, username):
    with app.app_context():
        try:
            summary = run_trading_simulation(filepath)
            tx_hash = blockchain.add_transaction({
                'source': 'AlgorithmRun', 'user': username, 'summary': summary
            })
            socketio.emit('file_update', {
                'status': 'processed', 'filename': os.path.basename(filepath),
                'username': username, 'summary': summary, 'tx_hash': tx_hash
            })
        except Exception as e:
            socketio.emit('file_update', {'status': 'error', 'error': str(e)})

@app.route('/api/dashboard-data')
@login_required
def dashboard_data():
    chain = blockchain.get_chain_data()
    return jsonify({
        'blockchain': {'length': len(chain), 'chain': chain, 'pending_tx': len(blockchain.pending_transactions)},
        'mqtt_log': list(MQTT_LOG_CACHE)
    })

@app.route('/mine', methods=['POST'])
@login_required
def mine():
    blk = blockchain.mine_pending_transactions()
    if blk:
        socketio.emit('blockchain_update', blk)
        return jsonify({'success': 'Mined', 'block': blk}), 200
    return jsonify({'error': 'Nothing to mine'}), 400

@app.route('/block/<int:index>')
@login_required
def get_block_detail(index):
    if 0 <= index < len(blockchain.chain):
        return jsonify(blockchain.chain[index].to_dict())
    return jsonify({'error': 'Invalid index'}), 404

# --- Socket Events ---
@socketio.on('connect')
def on_connect():
    if current_user.is_authenticated:
        emit('welcome', {'message': f'Welcome {current_user.id}'})

# --- Main ---
if __name__ == '__main__':
    Thread(target=background_data_simulator, daemon=True).start()
    Thread(target=background_bid_monitor, daemon=True).start()
    setup_mqtt_client()
    
    # Host 0.0.0.0 allows external connections
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)