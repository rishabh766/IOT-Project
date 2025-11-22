#!/usr/bin/env python3
"""
Peer-to-Peer (P2P) ONLY Energy Trading Simulation

This script implements a community-level P2P market for an
"islanded" community with NO connection to the main grid.

### MODIFICATION ###
This version now reads the 'type' for each household (prosumer, consumer)
and assigns assets (batteries) based on that type.
This creates a realistic market of buyers (no battery) and sellers (has battery).

### NEW FEATURES ###
- User-defined Base Price option.
- Tracking of Clearing Price.
- Tracking of Member-wise Profits.
- ROBUST DATA LOADING: Handles missing prices.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import matplotlib.dates as mdates

# --- Simulation Parameters (Tune These) ---

# 1. Data Configuration
DATASET_DIR = '../generate_dataset/example_dataset'
PRICE_FILE = 'prices.csv'

# 2. User Options
# Set this to a float (e.g., 0.15) to override grid prices with a constant base price.
# Set to None to use the prices.csv file.
USER_BASE_PRICE = 7.0  # Example: 7.0 currency units

# 3. Algorithm Parameters (from paper)
V = 50.0
BETA = 0.8

# 4. P2P Market Parameters
P2P_BUY_FACTOR = 0.90   # What buyers pay (relative to Base/Grid Price)
P2P_SELL_FACTOR = 0.85  # What sellers get (relative to Base/Grid Price)

# 5. Physical Battery Parameters (Assigned to Prosumers/Balanced)
B_MAX_KW = 5.0
B_CAPACITY_KWH = 50.0
THETA = B_CAPACITY_KWH * 0.6

# --- End of Parameters ---

def load_all_data(dataset_dir, price_file):
    """Loads all household data, price data, and summary file."""
    print(f"Loading all data from '{dataset_dir}'...")
    dir_path = Path(dataset_dir)
    pr_file = dir_path / price_file
    summary_file = dir_path / 'dataset_summary.json'

    if not summary_file.exists():
        print(f"Error: Summary file not found at {summary_file}.")
        print("Please run `generate_dataset.py` first.")
        return None, None, None, None

    # Check if price file exists
    df_pr = None
    if pr_file.exists():
        df_pr = pd.read_csv(pr_file, parse_dates=['timestamp'])
        print(" ✓ Loaded prices.csv")
    else:
        print(" ⚠ Warning: prices.csv not found. Will use USER_BASE_PRICE as default.")

    # ### MODIFICATION: Load household types from summary ###
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # Create a simple dictionary {'HH001': 'prosumer', 'HH002': 'consumer'}
    household_types = {hh['household_id']: hh['type'] for hh in summary['households']}
    household_ids = list(household_types.keys())
    n_households = len(household_ids)
    interval_minutes = summary['configuration']['interval_minutes']

    print(f" Found {n_households} households.")

    all_hh_data = []
    for hh_id in household_ids:
        hh_file = dir_path / f"{hh_id}.csv"
        if not hh_file.exists():
             print(f"Error: Household file {hh_file} missing.")
             continue

        df_hh = pd.read_csv(hh_file, parse_dates=['timestamp'])

        # Merge price data if available
        if df_pr is not None:
            df = pd.merge(df_hh, df_pr, on='timestamp')
        else:
            # If no price file, fill with USER_BASE_PRICE or 0.0
            df = df_hh.copy()
            fill_price = USER_BASE_PRICE if USER_BASE_PRICE is not None else 0.0
            df['price'] = fill_price

        df = df.rename(columns={'load_kW': 'load_kw', 'solar_kW': 'solar_kw'})
        df = df.set_index('timestamp')
        all_hh_data.append(df)

    print(f" ✓ Loaded data for {n_households} households.")

    # Return the household_types dictionary
    return all_hh_data, household_ids, household_types, interval_minutes

def run_p2p_simulation(all_hh_data, household_ids, household_types, n_households, interval_minutes, V, B_MAX_KW, B_CAPACITY_KWH, THETA, base_price=None):
    """
    Runs the P2P-ONLY simulation with asset heterogeneity.
    Supports a fixed base_price override.
    """
    print("Running P2P-Only (No Grid) simulation with asset heterogeneity...")
    if base_price is not None:
        print(f" -> Using User-Given Base Price: {base_price}")
    else:
        print(f" -> Using Dynamic Prices from data file.")

    energy_factor = interval_minutes / 60.0
    b_max_energy = B_MAX_KW * energy_factor

    # ### MODIFICATION: Initialize assets based on type ###
    B_current_kwh = {} # Use a dictionary to store battery levels
    has_battery = {}   # Use a dictionary to track who has a battery

    for hh_id in household_ids:
        hh_type = household_types[hh_id]
        if hh_type == 'consumer':
            has_battery[hh_id] = False
            B_current_kwh[hh_id] = 0.0 
        else: # Prosumers and Balanced get a battery
            has_battery[hh_id] = True
            B_current_kwh[hh_id] = B_CAPACITY_KWH / 2.0 # Start at 50%

    print(f" ✓ Assigned batteries to prosumers/balanced, not to consumers.")
    # --- End of Modification ---

    # History lists
    B_history = {hh_id: [] for hh_id in household_ids}
    b_history_kw = {hh_id: [] for hh_id in household_ids}

    # NEW: Member-wise profit tracking
    member_profits = {hh_id: [] for hh_id in household_ids}

    agg_results = {
        'p2p_buy_price': [],
        'p2p_sell_price': [],
        'clearing_price': [],  # NEW
        'total_load_kw': [], 'total_solar_kw': [],
        'total_p2p_trade_kw': [],
        'total_unmet_demand_kw': [],
        'total_excess_supply_kw': [],
        'total_profit_p2p': [],
        'total_solar_curtailed_kw': []
    }

    n_timesteps = len(all_hh_data[0])

    for t_idx in range(n_timesteps):
        # 1. Determine Prices
        if base_price is not None:
            P_grid_buy_t = base_price
        else:
            P_grid_buy_t = all_hh_data[0]['price'].iloc[t_idx]

        P_p2p_buy_t = P_grid_buy_t * P2P_BUY_FACTOR
        P_p2p_sell_t = P_grid_buy_t * P2P_SELL_FACTOR

        # Clearing price (mid-point)
        clearing_price_t = (P_p2p_buy_t + P_p2p_sell_t) / 2.0

        g_intentions_energy = {} # Use dicts
        b_actions_energy = {}    # Use dicts

        total_p2p_demand = 0.0
        total_p2p_supply = 0.0
        total_load_t = 0.0
        total_solar_t = 0.0
        total_curtailment_t = 0.0

        # 2. Determine Intentions (Prosumer decision making)
        for h in range(n_households):
            hh_id = household_ids[h]
            B_t = B_current_kwh[hh_id]
            B_history[hh_id].append(B_t)

            row = all_hh_data[h].iloc[t_idx]
            d_t_energy = row['load_kw'] * energy_factor
            r_t_energy = row['solar_kw'] * energy_factor

            total_load_t += row['load_kw']
            total_solar_t += row['solar_kw']

            # ### MODIFICATION: New logic based on 'has_battery' ###
            b_actual_energy = 0.0

            if has_battery[hh_id]:
                # Prosumer Logic
                alpha_t = V * P_p2p_buy_t + B_t - THETA
                gamma_t = V * P_p2p_sell_t + B_t - THETA

                b_ideal_energy = 0.0
                if gamma_t >= 0:
                    b_ideal_energy = b_max_energy # Sell
                elif alpha_t <= 0:
                    b_ideal_energy = -b_max_energy # Buy
                elif alpha_t > 0 > gamma_t:
                    b_ideal_energy = min(b_max_energy, d_t_energy) # Self-supply

                charge_limit = -(B_CAPACITY_KWH - B_t)
                discharge_limit = B_t
                b_actual_energy = np.clip(b_ideal_energy, 
                                        max(charge_limit, -b_max_energy), 
                                        min(discharge_limit, b_max_energy))
            else:
                # Consumer Logic
                b_actual_energy = 0.0

            # Net energy need g(t) = d(t) - r(t) - b(t)
            g_t_energy = d_t_energy - r_t_energy - b_actual_energy

            g_intentions_energy[hh_id] = g_t_energy
            b_actions_energy[hh_id] = b_actual_energy
            b_history_kw[hh_id].append(b_actual_energy / energy_factor)

            if g_t_energy > 0:
                total_p2p_demand += g_t_energy
            elif g_t_energy < 0:
                total_p2p_supply += abs(g_t_energy)
            # --- End of Modification ---

        # 3. Market Clearing (Pro-rata)
        total_p2p_trade_kwh = min(total_p2p_demand, total_p2p_supply)
        buy_ratio = total_p2p_trade_kwh / total_p2p_demand if total_p2p_demand > 0 else 0
        sell_ratio = total_p2p_trade_kwh / total_p2p_supply if total_p2p_supply > 0 else 0

        total_unmet_demand_kwh = 0.0
        total_excess_supply_kwh = 0.0
        total_p2p_profit_t = 0.0

        # 4. Settlement & Battery Updates
        for h in range(n_households):
            hh_id = household_ids[h]
            g_t_energy = g_intentions_energy[hh_id]
            b_t_energy = b_actions_energy[hh_id]

            hh_profit = 0.0

            if g_t_energy > 0:
                # Buyer
                p2p_buy_kwh = -(g_t_energy * buy_ratio)
                unmet_demand_kwh = -(g_t_energy * (1 - buy_ratio))

                cost_p2p = abs(p2p_buy_kwh) * P_p2p_buy_t
                hh_profit -= cost_p2p # Cost is negative profit
                total_p2p_profit_t -= cost_p2p

                total_unmet_demand_kwh += abs(unmet_demand_kwh)

            elif g_t_energy < 0:
                # Seller
                p2p_sell_kwh = abs(g_t_energy * sell_ratio)
                excess_supply_kwh = abs(g_t_energy * (1 - sell_ratio))

                rev_p2p = p2p_sell_kwh * P_p2p_sell_t
                hh_profit += rev_p2p # Revenue is positive profit
                total_p2p_profit_t += rev_p2p

                total_excess_supply_kwh += abs(excess_supply_kwh)

            # Track Member Profit
            member_profits[hh_id].append(hh_profit)

            # Battery Update Logic
            if has_battery[hh_id]:
                B_t = B_current_kwh[hh_id]
                B_next_kwh = B_t - b_actions_energy[hh_id]
                B_current_kwh[hh_id] = np.clip(B_next_kwh, 0, B_CAPACITY_KWH)

            else:
                pass

        agg_results['p2p_buy_price'].append(P_p2p_buy_t)
        agg_results['p2p_sell_price'].append(P_p2p_sell_t)
        agg_results['clearing_price'].append(clearing_price_t)
        agg_results['total_load_kw'].append(total_load_t)
        agg_results['total_solar_kw'].append(total_solar_t)
        agg_results['total_p2p_trade_kw'].append(total_p2p_trade_kwh / energy_factor)
        agg_results['total_unmet_demand_kw'].append(total_unmet_demand_kwh / energy_factor)
        agg_results['total_excess_supply_kw'].append(total_excess_supply_kwh / energy_factor)
        agg_results['total_profit_p2p'].append(total_p2p_profit_t)
        agg_results['total_solar_curtailed_kw'].append(total_curtailment_t)

    print(" ✓ P2P-Only simulation complete.")
    df_agg = pd.DataFrame(agg_results, index=all_hh_data[0].index)
    df_agg = df_agg.rename(columns={'total_p2p_trade_kw': 'total_p2p_trade_kwh'})

    # Convert member profits to DataFrame
    df_member_profits = pd.DataFrame(member_profits, index=all_hh_data[0].index)

    return df_agg, df_member_profits

def plot_requested_graphs(df_agg, df_member_profits):
    """
    Generates the plots requested by the user, including Member Profits.
    """
    print("Generating requested plots...")

    # We'll do 3 subplots now
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True)
    fig.suptitle("P2P Community Simulation Analysis", fontsize=16)

    # --- Plot 1: Total P2P Energy Traded per Hour ---
    ax1 = axes[0]
    df_agg['total_p2p_trade_kwh'].plot(kind='line', ax=ax1, 
                                     label='P2P Energy Traded', 
                                     color='blue', alpha=0.7)
    ax1.set_title('Total P2P Energy Traded per Hour')
    ax1.set_ylabel('Energy Traded (kWh)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

    # --- Plot 2: Hourly Price Variations (Clearing Price) ---
    ax2 = axes[1]
    # Plot the explicitly calculated clearing price
    df_agg['clearing_price'].plot(ax=ax2, label='P2P Clearing Price', color='purple', 
                                           linestyle='-', marker=None)

    # Also show Buy/Sell bands for context
    df_agg['p2p_buy_price'].plot(ax=ax2, label='Buy Price', color='red', linestyle=':', alpha=0.5)
    df_agg['p2p_sell_price'].plot(ax=ax2, label='Sell Price', color='green', linestyle=':', alpha=0.5)

    ax2.set_title('Hourly Price Variations (Clearing, Buy, Sell)')
    ax2.set_ylabel('Price ($/kWh)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 3: Member-wise Profits (Cumulative) ---
    ax3 = axes[2]
    # Plot cumulative profit for top 5 performing members to avoid clutter
    # or just all of them if N is small
    cumulative_profits = df_member_profits.cumsum()

    # Select a few representative households if too many
    if len(cumulative_profits.columns) > 10:
        # Plot Top 3 and Bottom 3
        final_vals = cumulative_profits.iloc[-1]
        top_3 = final_vals.nlargest(3).index
        bottom_3 = final_vals.nsmallest(3).index
        cols_to_plot = list(top_3) + list(bottom_3)
        cumulative_profits[cols_to_plot].plot(ax=ax3, alpha=0.8)
    else:
        cumulative_profits.plot(ax=ax3, alpha=0.8)

    ax3.set_title('Cumulative Member Profits Over Time')
    ax3.set_ylabel('Net Profit ($)')
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Format the x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')

    plt.xlabel('Timestamp')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print(" ✓ Plots generated. Displaying results...")
    # plt.show() # Commented out to avoid blocking in some environments

def main():
    """Main execution function."""

    # 1. Load data
    all_hh_data, hh_ids, household_types, interval_minutes = load_all_data(DATASET_DIR, PRICE_FILE)

    if all_hh_data is None:
        return

    n_households = len(all_hh_data)

    # 2. Get User Input for Base Price (Optional)
    base_price_input = USER_BASE_PRICE

    # 3. Run simulation
    df_agg, df_member_profits = run_p2p_simulation(
        all_hh_data, hh_ids, household_types, n_households, interval_minutes,
        V, B_MAX_KW, B_CAPACITY_KWH, THETA, 
        base_price=base_price_input
    )

    # 4. Plot results
    plot_requested_graphs(df_agg, df_member_profits)

    # --- Print Final Summary ---
    total_profit = df_agg['total_profit_p2p'].cumsum().iloc[-1]
    energy_factor = interval_minutes / 60.0
    total_p2p = df_agg['total_p2p_trade_kwh'].sum()
    total_unmet = df_agg['total_unmet_demand_kw'].sum() * energy_factor
    total_excess = df_agg['total_excess_supply_kw'].sum() * energy_factor

    print(f"\n--- P2P-Only Simulation Summary ---")
    print(f"Base Price Used: {base_price_input if base_price_input else 'Dynamic Grid Price'}")
    print(f"Total Community P2P Profit/Loss: ${total_profit:.2f}")
    print(f"\nTotal Energy (Community):")
    print(f" - P2P Community Trade: {total_p2p:.1f} kWh")
    print(f" - Total Unmet Demand: {total_unmet:.1f} kWh")
    print(f" - Total Excess Supply (Not Sold): {total_excess:.1f} kWh")

    print(f"\n--- Member-wise Financials (Top 5 Winners) ---")
    final_profits = df_member_profits.sum().sort_values(ascending=False)
    print(final_profits.head(5))
    print("...")
    print(final_profits.tail(3))

if __name__ == "__main__":
    main()
