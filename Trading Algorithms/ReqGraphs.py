#!/usr/bin/env python3
"""
Peer-to-Peer (P2P) ONLY Energy Trading Simulation

This script implements a community-level P2P market for an
"islanded" community with NO connection to the main grid.

### MODIFICATION ###
This version now reads the 'type' for each household (prosumer, consumer)
and assigns assets (batteries) based on that type.
This creates a realistic market of buyers (no battery) and sellers (has battery).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import matplotlib.dates as mdates 

# --- Simulation Parameters (Tune These) ---

# 1. Data Configuration
DATASET_DIR = 'example_dataset'
PRICE_FILE = 'prices.csv' 

# 2. Algorithm Parameters (from paper)
V = 50.0        
BETA = 0.8      

# 3. P2P Market Parameters
P2P_BUY_FACTOR = 0.90  # What buyers pay
P2P_SELL_FACTOR = 0.85 # What sellers get

# 4. Physical Battery Parameters (Assigned to Prosumers/Balanced)
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
    
    if not pr_file.exists() or not summary_file.exists():
        print(f"Error: Price or summary file not found.")
        print("Please run `generate_dataset_with_prices.py` first.")
        return None, None, None, None

    # ### MODIFICATION: Load household types from summary ###
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Create a simple dictionary {'HH001': 'prosumer', 'HH002': 'consumer'}
    household_types = {hh['household_id']: hh['type'] for hh in summary['households']}
    household_ids = list(household_types.keys())
    
    n_households = len(household_ids)
    interval_minutes = summary['configuration']['interval_minutes']
    print(f"  Found {n_households} households.")

    df_pr = pd.read_csv(pr_file, parse_dates=['timestamp'])
    
    all_hh_data = []
    for hh_id in household_ids:
        hh_file = dir_path / f"{hh_id}.csv"
        df_hh = pd.read_csv(hh_file, parse_dates=['timestamp'])
        df = pd.merge(df_hh, df_pr, on='timestamp')
        df = df.rename(columns={'load_kW': 'load_kw', 'solar_kW': 'solar_kw'})
        df = df.set_index('timestamp')
        all_hh_data.append(df)
    
    print(f"  ✓ Loaded data for {n_households} households.")
    # Return the household_types dictionary
    return all_hh_data, household_ids, household_types, interval_minutes


def run_p2p_simulation(all_hh_data, household_ids, household_types, n_households, interval_minutes, V, B_MAX_KW, B_CAPACITY_KWH, THETA):
    """
    Runs the P2P-ONLY simulation with asset heterogeneity.
    """
    print("Running P2P-Only (No Grid) simulation with asset heterogeneity...")
    
    energy_factor = interval_minutes / 60.0 
    b_max_energy = B_MAX_KW * energy_factor

    # ### MODIFICATION: Initialize assets based on type ###
    B_current_kwh = {} # Use a dictionary to store battery levels
    has_battery = {}   # Use a dictionary to track who has a battery
    
    for hh_id in household_ids:
        hh_type = household_types[hh_id]
        if hh_type == 'consumer':
            has_battery[hh_id] = False
            B_current_kwh[hh_id] = 0.0 # No battery, so 0 charge
        else: # Prosumers and Balanced get a battery
            has_battery[hh_id] = True
            B_current_kwh[hh_id] = B_CAPACITY_KWH / 2.0 # Start at 50%
            
    print(f"  ✓ Assigned batteries to prosumers/balanced, not to consumers.")
    # --- End of Modification ---

    # History lists now need to be dictionaries
    B_history = {hh_id: [] for hh_id in household_ids}
    b_history_kw = {hh_id: [] for hh_id in household_ids}
    
    agg_results = {
        'p2p_buy_price': [],  
        'p2p_sell_price': [], 
        'total_load_kw': [], 'total_solar_kw': [],
        'total_p2p_trade_kw': [],
        'total_unmet_demand_kw': [], 
        'total_excess_supply_kw': [],
        'total_profit_p2p': [],
        'total_solar_curtailed_kw': []
    }
    
    n_timesteps = len(all_hh_data[0])
    
    for t_idx in range(n_timesteps):
        t = all_hh_data[0].index[t_idx]
        P_grid_buy_t = all_hh_data[0]['price'].iloc[t_idx]
        
        P_p2p_buy_t = P_grid_buy_t * P2P_BUY_FACTOR
        P_p2p_sell_t = P_grid_buy_t * P2P_SELL_FACTOR
        
        g_intentions_energy = {} # Use dicts
        b_actions_energy = {}    # Use dicts
        total_p2p_demand = 0.0
        total_p2p_supply = 0.0
        total_load_t = 0.0
        total_solar_t = 0.0
        total_curtailment_t = 0.0

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
                # This is a Prosumer/Balanced household, run the ETA algorithm
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
                # This is a Consumer. They have no battery.
                b_actual_energy = 0.0
            
            # Net energy need g(t) = d(t) - r(t) - b(t)
            # We must account for their own solar first!
            g_t_energy = d_t_energy - r_t_energy - b_actual_energy
            
            g_intentions_energy[hh_id] = g_t_energy
            b_actions_energy[hh_id] = b_actual_energy
            b_history_kw[hh_id].append(b_actual_energy / energy_factor)

            if g_t_energy > 0:
                total_p2p_demand += g_t_energy
            elif g_t_energy < 0:
                total_p2p_supply += abs(g_t_energy) 
            # --- End of Modification ---

        # This part will now have non-zero values for both!
        total_p2p_trade_kwh = min(total_p2p_demand, total_p2p_supply)
        
        buy_ratio = total_p2p_trade_kwh / total_p2p_demand if total_p2p_demand > 0 else 0
        sell_ratio = total_p2p_trade_kwh / total_p2p_supply if total_p2p_supply > 0 else 0
        
        total_unmet_demand_kwh = 0.0
        total_excess_supply_kwh = 0.0
        total_p2p_profit_t = 0.0
        
        for h in range(n_households):
            hh_id = household_ids[h]
            g_t_energy = g_intentions_energy[hh_id]
            b_t_energy = b_actions_energy[hh_id]
            
            if g_t_energy > 0: 
                p2p_buy_kwh = -(g_t_energy * buy_ratio)
                unmet_demand_kwh = -(g_t_energy * (1 - buy_ratio))
                cost_p2p = abs(p2p_buy_kwh) * P_p2p_buy_t
                total_p2p_profit_t -= cost_p2p
                total_unmet_demand_kwh += abs(unmet_demand_kwh)
                
            elif g_t_energy < 0:
                p2p_sell_kwh = abs(g_t_energy * sell_ratio)
                excess_supply_kwh = abs(g_t_energy * (1 - sell_ratio)) 
                rev_p2p = p2p_sell_kwh * P_p2p_sell_t
                total_p2p_profit_t += rev_p2p
                total_excess_supply_kwh += abs(excess_supply_kwh)

            # Battery update logic
            if has_battery[hh_id]:
                r_t_energy = all_hh_data[h].iloc[t_idx]['solar_kw'] * energy_factor
                B_t = B_current_kwh[hh_id]
                # Update B(t+1) = B(t) - b(t) + [r(t) - d(t)]
                # Simpler: B(t+1) = B(t) - b(t) - g(t)_after_solar
                # The net energy g_t_energy = d_t - r_t - b_t
                # The energy surplus/deficit *after* solar is (d_t - r_t)
                # Let's rewrite the logic for clarity.
                
                # g_t_energy = d_t - r_t - b_t
                # if g_t > 0, we bought from P2P/grid.
                # if g_t < 0, we sold to P2P/grid.
                
                # The energy *not* handled by the battery is (d_t - r_t)
                net_load_energy = d_t_energy - r_t_energy
                
                # The battery's contribution is b_t
                # Let's re-think the update rule.
                # B(t+1) = B(t) - b(t) + r(t)
                # This was wrong. b(t) is *from* the battery.
                # Let's trace the energy
                
                # 1. Solar `r_t` is generated.
                # 2. Load `d_t` consumes `r_t`.
                net_load = d_t_energy - r_t_energy
                
                if net_load > 0: # Need energy
                    # Battery discharges `b_t` to cover `net_load`.
                    # b_t is positive (discharge)
                    # g_t = net_load - b_t
                    
                    # This logic is confusing. Let's restart the simulation loop logic.
                    # Go back to the original `Plotting.py` logic.
                    # b_actual_energy: + is discharge, - is charge
                    # g_t_energy = d_t_energy - b_actual_energy (This is net *load*)
                    # B(t+1) = B(t) - b(t) + r(t) (This is net *battery*)
                    
                    # Let's use the old, correct logic
                    # B(t+1) = B(t) - b(t_energy) + r(t_energy)
                    # This assumes b(t) is *charge*
                    # In `Plotting.py`: `b_history_kw` # Positive = Discharge
                    # `B_next_kwh = B_t - b_actual_energy + r_t_energy`
                    # This is correct. `b_actual_energy` is positive for discharge (B_t decreases)
                    # `b_actual_energy` is negative for charge (B_t increases)
                    
                    # Back to this script's loop
                    B_t = B_current_kwh[hh_id]
                    B_next_kwh = B_t - b_actions_energy[hh_id] + r_t_energy
                    
                    solar_curtailed_energy = max(0, B_next_kwh - B_CAPACITY_KWH)
                    total_curtailment_t += (solar_curtailed_energy / energy_factor)
                    B_current_kwh[hh_id] = np.clip(B_next_kwh, 0, B_CAPACITY_KWH)
                
                else:
                    # Consumer: no battery update needed
                    # We must still check for their curtailment
                    # If r_t > d_t, they curtailed (r_t - d_t)
                    if r_t_energy > d_t_energy:
                         # This consumer wasted solar
                         curtailed = r_t_energy - d_t_energy
                         total_curtailment_t += (curtailed / energy_factor)


        agg_results['p2p_buy_price'].append(P_p2p_buy_t)
        agg_results['p2p_sell_price'].append(P_p2p_sell_t)
        agg_results['total_load_kw'].append(total_load_t)
        agg_results['total_solar_kw'].append(total_solar_t)
        agg_results['total_p2p_trade_kw'].append(total_p2p_trade_kwh / energy_factor)
        agg_results['total_unmet_demand_kw'].append(total_unmet_demand_kwh / energy_factor)
        agg_results['total_excess_supply_kw'].append(total_excess_supply_kwh / energy_factor)
        agg_results['total_profit_p2p'].append(total_p2p_profit_t)
        agg_results['total_solar_curtailed_kw'].append(total_curtailment_t)

    print("  ✓ P2P-Only simulation complete.")
    
    df_agg = pd.DataFrame(agg_results, index=all_hh_data[0].index)
    
    df_agg = df_agg.rename(columns={'total_p2p_trade_kw': 'total_p2p_trade_kwh'})
    
    return df_agg

def plot_requested_graphs(df_agg):
    """
    Generates the two specific plots requested by the user.
    This function is unchanged, but will now plot real data.
    """
    print("Generating requested plots...")
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
    fig.suptitle("P2P Community Simulation Analysis", fontsize=16)

    # --- Plot 1: Total P2P Energy Traded per Hour ---
    ax1 = axes[0]
    # We will use a bar plot for the 7-day (168 hour) data
    # A line plot is better for this much data
    df_agg['total_p2p_trade_kwh'].plot(kind='line', ax=ax1, 
                                        label='P2P Energy Traded', 
                                        color='blue', alpha=0.7)
    
    ax1.set_title('Total P2P Energy Traded per Hour')
    ax1.set_ylabel('Energy Traded (kWh)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # --- Plot 2: Hourly Price Variations (Market Clearing Price) ---
    ax2 = axes[1]
    
    # Calculate the average P2P price to plot as a single line
    df_agg['p2p_market_price'] = (df_agg['p2p_buy_price'] + df_agg['p2p_sell_price']) / 2
    
    df_agg['p2p_market_price'].plot(ax=ax2, label='P2P Market Price', color='purple', 
                                    linestyle='-', marker=None) # Line plot
    
    ax2.set_title('Hourly Price Variations (Market Clearing Price vs. Hour)')
    ax2.set_ylabel('P2P Price ($/kWh)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Format the x-axis for a multi-day plot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
    
    plt.xlabel('Timestamp')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    print("  ✓ Plots generated. Displaying results...")
    plt.show()


def main():
    """Main execution function."""
    
    # 1. Load data
    # ### MODIFICATION: Load new `household_types` dict ###
    all_hh_data, hh_ids, household_types, interval_minutes = load_all_data(DATASET_DIR, PRICE_FILE)
    
    if all_hh_data is None:
        return 
        
    n_households = len(all_hh_data)
        
    # 2. Run simulation
    # ### MODIFICATION: Pass new variables to simulation ###
    df_agg = run_p2p_simulation(
        all_hh_data, hh_ids, household_types, n_households, interval_minutes, 
        V, B_MAX_KW, B_CAPACITY_KWH, THETA
    )
    
    # 3. Plot results
    plot_requested_graphs(df_agg)
    
    # --- Print Final Summary ---
    total_profit = df_agg['total_profit_p2p'].cumsum().iloc[-1]
    
    energy_factor = interval_minutes / 60.0
    total_p2p = df_agg['total_p2p_trade_kwh'].sum() 
    total_unmet = df_agg['total_unmet_demand_kw'].sum() * energy_factor
    total_excess = df_agg['total_excess_supply_kw'].sum() * energy_factor
    total_curtailed = df_agg['total_solar_curtailed_kw'].sum() * energy_factor

    print(f"\n--- P2P-Only Simulation Summary ---")
    print(f"Total Community P2P Profit/Loss: ${total_profit:.2f}")
    print(f"\nTotal Energy (Community):")
    print(f"  - P2P Community Trade: {total_p2p:.1f} kWh")
    print(f"  - Total Unmet Demand: {total_unmet:.1f} kWh")
    print(f"  - Total Excess Supply (Not Sold): {total_excess:.1f} kWh")
    print(f"  - Total Solar Wasted (Battery Full or No Battery): {total_curtailed:.1f} kWh")


if __name__ == "__main__":
    main()