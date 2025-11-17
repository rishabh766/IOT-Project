import pandas as pd
import json
import glob
import os

# --- Parameters (from tradingalgo.ipynb) ---
# BASE_PRICE is now passed as an argument.
ALPHA = 5.0  # Solar abundance price sensitivity
BETA = 0.125  # Buyer discount below grid


def run_trading_simulation(uploaded_filepath, base_price=7.0):
    """
    Runs the trading algorithm based on the uploaded file and other
    CSVs in the 'uploads' directory.

    :param uploaded_filepath: Path to the uploaded CSV file.
    :param base_price: The grid price baseline (₹/kWh) to use for the auction.
    :return: Dictionary containing the simulation summary.
    """

    print(f"Starting trading simulation for: {uploaded_filepath} with BASE_PRICE={base_price}")

    # Use the directory of the uploaded file
    upload_dir = os.path.dirname(uploaded_filepath)

    # 1. Gather all CSVs from the 'uploads' folder
    files = sorted(glob.glob(os.path.join(upload_dir, 'HH*.csv')))
    if not files:
        raise ValueError("No HH*.csv files found in uploads directory.")

    all_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Algorithm: Loaded {len(files)} CSV files.")

    # 2. Mock household info (In a real app, this would be in your database)
    # This is a simplified version of your 'dataset_summary.json' logic
    # We'll create a mock info dict based on household IDs found
    hh_ids_found = all_data['household_id'].unique()
    hh_info = {}
    for i, hhid in enumerate(hh_ids_found):
        # Mocking data: assume first 3 are prosumers, rest are consumers
        # And assign a random-ish pv_kw
        if i < 3:
            hh_info[hhid] = {'household_id': hhid, 'type': 'prosumer', 'pv_kw': 3.0 + (i * 0.1)}
        else:
            hh_info[hhid] = {'household_id': hhid, 'type': 'consumer', 'pv_kw': 1.0 + (i * 0.1)}

    # 3. Run the auction clearing algorithm
    results = []
    trades_log = []

    # Group by each timestamp to run the auction
    for ts, group in all_data.groupby('timestamp'):
        bids, asks = [], []
        for _, row in group.iterrows():
            hhid, load, solar = row['household_id'], row['load_kW'], row['solar_kW']

            # Skip if data is missing
            if hhid not in hh_info:
                continue

            pv_size = hh_info[hhid]['pv_kw']
            solar_norm = min(solar / pv_size, 1.0) if pv_size > 0 else 0
            surplus = solar - load

            if surplus >= 0:  # Seller (Prosumer)
                # Use base_price from argument
                ask_price = base_price - (ALPHA * solar_norm)
                asks.append({'hhid': hhid, 'qty': surplus, 'price': ask_price})
            else:  # Buyer (Consumer or Prosumer in deficit)
                # Use base_price from argument
                bid_price = base_price - BETA
                bids.append({'hhid': hhid, 'qty': -surplus, 'price': bid_price})

        # Sort bids (high to low) and asks (low to high)
        bids_sorted = sorted(bids, key=lambda x: -x['price'])
        asks_sorted = sorted(asks, key=lambda x: x['price'])

        # Auction clearing
        traded_kwh = 0
        b_idx, a_idx = 0, 0
        n_trades_at_ts = 0

        while b_idx < len(bids_sorted) and a_idx < len(asks_sorted):
            buyer = bids_sorted[b_idx]
            seller = asks_sorted[a_idx]

            if buyer['price'] >= seller['price']:
                # Trade can happen
                trade_qty = min(buyer['qty'], seller['qty'])
                traded_kwh += trade_qty
                n_trades_at_ts += 1

                # Log individual trade
                trades_log.append({
                    'timestamp': ts,
                    'buyer': buyer['hhid'],
                    'seller': seller['hhid'],
                    'qty_kwh': trade_qty,
                    'price': (buyer['price'] + seller['price']) / 2  # Mid-point price
                })

                # Update quantities
                buyer['qty'] -= trade_qty
                seller['qty'] -= trade_qty

                if buyer['qty'] == 0:
                    b_idx += 1
                if seller['qty'] == 0:
                    a_idx += 1
            else:
                # No more trades possible
                break

        # Market Clearing Price (average of marginal prices or fallback)
        # Use base_price from argument
        clearing_price = (bids_sorted[b_idx - 1]['price'] + asks_sorted[a_idx - 1][
            'price']) / 2 if traded_kwh > 0 else base_price

        # Calculate savings
        # Use base_price from argument
        grid_cost = traded_kwh * base_price
        auction_cost = traded_kwh * clearing_price
        cost_savings = grid_cost - auction_cost

        results.append({
            'timestamp': ts,
            'clearing_price': clearing_price,
            'total_traded_kwh': traded_kwh,
            'cost_savings': cost_savings,
            'n_trades': n_trades_at_ts
        })

    # 4. Generate summary
    summary_df = pd.DataFrame(results)
    total_traded = summary_df['total_traded_kwh'].sum()
    total_savings = summary_df['cost_savings'].sum()
    # Ensure mean calculation handles cases where clearing_price might be base_price
    avg_price = summary_df[summary_df['clearing_price'] < base_price]['clearing_price'].mean()

    summary_output = {
        "files_processed": len(files),
        "total_kwh_traded": round(total_traded, 2),
        "total_cost_savings": f"₹{total_savings:,.2f}",
        # Fallback if no trades occurred below base price
        "avg_p2p_price": f"₹{round(avg_price, 2) if not pd.isna(avg_price) else base_price:.2f}",
        "total_trades": len(trades_log)
    }

    print(f"Algorithm finished. Summary: {summary_output}")
    return summary_output