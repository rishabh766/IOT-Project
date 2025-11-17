# -*- coding: utf-8 -*-
"""
Blockchain Service Module for P2P Energy Trading.

Features:
1.  Password Hashing: For future user authentication.
2.  Data Hashing: SHA-256 for trade integrity.
3.  Blockchain: PoW ledger to record energy trades.
4.  Persistence: Saves/Loads chain from JSON log file.
"""

import bcrypt
import hashlib
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configuration
CHAIN_FILE = 'ledger.json'  # File to store the blockchain


# =============================================================================
# Part 1: Security & Hashing
# =============================================================================

def hash_password(plain_text_password: str) -> str:
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(plain_text_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def check_password(plain_text_password: str, hashed_password: str) -> bool:
    """Verifies a password against a hash."""
    try:
        return bcrypt.checkpw(plain_text_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except (ValueError, TypeError):
        return False


def calculate_sha256_hash(data_string: str) -> str:
    """Standard SHA-256 hashing."""
    sha = hashlib.sha256()
    sha.update(data_string.encode('utf-8'))
    return sha.hexdigest()


def create_trade_hash(trade_data: Dict[str, Any]) -> str:
    """Creates a deterministic hash for a trade/energy transaction."""
    # Sort keys to ensure consistency
    ordered_string = json.dumps(trade_data, sort_keys=True)
    return calculate_sha256_hash(ordered_string)


# =============================================================================
# Part 2: Blockchain Classes
# =============================================================================

class Block:
    """A single block in the blockchain."""

    def __init__(self, index: int, timestamp: str, transactions: List[Dict], previous_hash: str, nonce: int = 0,
                 hash: str = None):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        # If loading from file, use existing hash, else compute new
        self.hash = hash if hash else self.compute_hash()

    def compute_hash(self) -> str:
        """Computes the hash of the entire block."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return calculate_sha256_hash(block_string)

    def to_dict(self):
        """Returns a dictionary representation of the block for JSON serialization."""
        return self.__dict__


class Blockchain:
    """Manages the chain of blocks, persistence, and mining."""

    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.difficulty = difficulty
        self.difficulty_prefix = '0' * difficulty

        # Load existing chain from disk or create genesis block
        if os.path.exists(CHAIN_FILE):
            self.load_chain()
        else:
            self.create_genesis_block()

    def create_genesis_block(self):
        """Creates the first block (Block 0) in the chain."""
        print("Creating Genesis Block...")
        genesis_block = Block(0, str(datetime.now()), [], "0")
        self.chain.append(genesis_block)
        self.save_chain()

    def get_last_block(self) -> Block:
        """Helper to get the most recent block."""
        return self.chain[-1]

    def add_transaction(self, trade: Dict[str, Any]):
        """
        Adds a new trade to the list of pending transactions.
        Called by app.py when MQTT message is received.
        """
        # Add a timestamp if not present
        if 'timestamp' not in trade:
            trade['timestamp'] = str(datetime.now())

        # Generate a transaction ID (tx_hash) if not present
        if 'tx_hash' not in trade:
            trade['tx_hash'] = create_trade_hash(trade)

        self.pending_transactions.append(trade)
        return trade['tx_hash']

    def proof_of_work(self, block: Block) -> str:
        """
SImple Proof-of-Work algorithm."""
        block.nonce = 0
        computed_hash = block.compute_hash()
        while not computed_hash.startswith(self.difficulty_prefix):
            block.nonce += 1
            computed_hash = block.compute_hash()
        return computed_hash

    def mine_pending_transactions(self) -> Optional[Dict]:
        """
        Mines a new block containing all pending transactions.
        Called by the background thread in app.py.
        """
        if not self.pending_transactions:
            return None  # No transactions to mine

        print(f"Mining block with {len(self.pending_transactions)} transactions...")
        last_block = self.get_last_block()

        new_block = Block(
            index=last_block.index + 1,
            timestamp=str(datetime.now()),
            transactions=self.pending_transactions[:],  # Copy list
            previous_hash=last_block.hash
        )

        # This is the CPU-intensive part
        new_hash = self.proof_of_work(new_block)
        new_block.hash = new_hash

        # Validate the new block before adding
        if self.is_valid_new_block(new_block, last_block):
            self.chain.append(new_block)
            self.pending_transactions = []  # Clear the pending pool
            self.save_chain()  # Persist to disk
            return new_block.to_dict()

        print("Mining failed: New block was not valid.")
        return None

    def is_valid_new_block(self, new_block: Block, prev_block: Block) -> bool:
        """Checks if a newly mined block is valid."""
        if prev_block.index + 1 != new_block.index:
            print("Block validation failed: Index mismatch.")
            return False
        if prev_block.hash != new_block.previous_hash:
            print("Block validation failed: Previous hash mismatch.")
            return False
        if not new_block.hash.startswith(self.difficulty_prefix):
            print("Block validation failed: Proof of work prefix invalid.")
            return False
        if new_block.hash != new_block.compute_hash():
            print("Block validation failed: Block hash and computed hash do not match.")
            return False
        return True

    def save_chain(self):
        """Saves the entire chain to the JSON file."""
        chain_data = [b.to_dict() for b in self.chain]
        try:
            with open(CHAIN_FILE, 'w') as f:
                json.dump(chain_data, f, indent=2)
        except Exception as e:
            print(f"Error saving blockchain to {CHAIN_FILE}: {e}")

    def load_chain(self):
        """Loads the chain from the JSON file on startup."""
        try:
            with open(CHAIN_FILE, 'r') as f:
                chain_data = json.load(f)
                self.chain = []
                for b_data in chain_data:
                    # Re-create Block objects from the dictionaries
                    block = Block(
                        b_data['index'], b_data['timestamp'], b_data['transactions'],
                        b_data['previous_hash'], b_data['nonce'], b_data['hash']
                    )
                    self.chain.append(block)
            print(f"Blockchain loaded from {CHAIN_FILE}. Length: {len(self.chain)}")
        except Exception as e:
            print(f"Error loading blockchain, starting fresh: {e}")
            self.create_genesis_block()

    def get_chain_data(self):
        """Returns the chain as a list of dictionaries for API response."""
        return [b.to_dict() for b in self.chain]