# P2P Energy Trading Dataset Generator - Complete Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Parameters](#parameters)
4. [Household Types](#household-types)
5. [Usage Examples](#usage-examples)
6. [Mathematical Details](#mathematical-details)
7. [Output Format](#output-format)
8. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
pip install pandas numpy pvlib
```

**Requirements:**

- Python 3.8+
- pandas >= 1.0.0
- numpy >= 1.17.0
- pvlib >= 0.9.0

---

## Quick Start

### Basic Usage

```bash
python generate_dataset_complete.py
```

This generates 10 households for 7 days with default balanced mix (33% prosumers, 33% consumers, 34% balanced).

### Customize with Python

```python
from generate_dataset_complete import generate_p2p_dataset

summary = generate_p2p_dataset(
    n_households=50,
    n_days=30,
    interval_minutes=60,
    prosumer_ratio=0.33,
    consumer_ratio=0.33,
    balanced_ratio=0.34
)
```

---

## Parameters

| Parameter          | Type      | Default               | Description                         |
| ------------------ | --------- | --------------------- | ----------------------------------- |
| `n_households`     | int       | 10                    | Number of households to generate    |
| `n_days`           | int       | 7                     | Simulation duration in days         |
| `interval_minutes` | int       | 60                    | Time resolution: 15, 30, or 60 only |
| `output_directory` | str       | 'p2p_simulation_data' | Folder to save CSV files            |
| `location`         | str/tuple | 'delhi'               | 'delhi' or (latitude, longitude)    |
| `prosumer_ratio`   | float     | 0.33                  | Fraction of prosumers (exporters)   |
| `consumer_ratio`   | float     | 0.33                  | Fraction of consumers (importers)   |
| `balanced_ratio`   | float     | 0.34                  | Fraction of balanced households     |
| `include_outages`  | bool      | True                  | Add random power outages            |
| `random_seed`      | int       | 42                    | For reproducibility                 |

**Constraint:** `prosumer_ratio + consumer_ratio + balanced_ratio = 1.0`

---

## Household Types

### Prosumers (Exporters)

- **Monthly consumption:** 80-120 kWh (LOW)
- **PV capacity:** 2.5-3.5 kW (LARGE)
- **Net balance:** Positive surplus
- **Role:** Export excess solar to grid/neighbors
- **Real-world examples:** Retired couple, remote home, commercial building with solar

### Consumers (Importers)

- **Monthly consumption:** 200-280 kWh (HIGH)
- **PV capacity:** 0.5-1.2 kW (SMALL)
- **Net balance:** Negative deficit
- **Role:** Import energy from neighbors/grid
- **Real-world examples:** Large family, apartment with limited roof space, heavy AC user

### Balanced (Flexible)

- **Monthly consumption:** 140-180 kWh (MEDIUM)
- **PV capacity:** 1.5-2.5 kW (MEDIUM)
- **Net balance:** Variable (~zero on average)
- **Role:** Can import or export depending on weather
- **Real-world examples:** Typical middle-class household, good solar coverage

---

## Usage Examples

### Scenario 1: Balanced Mix (Default)

```python
summary = generate_p2p_dataset(
    n_households=50,
    n_days=30,
    interval_minutes=60,
    prosumer_ratio=0.33,
    consumer_ratio=0.33,
    balanced_ratio=0.34
)
```

**Use for:** General P2P trading algorithm testing with realistic market diversity

### Scenario 2: High Solar Export

```python
summary = generate_p2p_dataset(
    n_households=50,
    n_days=30,
    interval_minutes=60,
    prosumer_ratio=0.50,    # 50% exporters
    consumer_ratio=0.25,    # 25% importers
    balanced_ratio=0.25     # 25% flexible
)
```

**Use for:** Low prices, high self-consumption studies, grid export analysis

### Scenario 3: High Demand

```python
summary = generate_p2p_dataset(
    n_households=50,
    n_days=30,
    interval_minutes=60,
    prosumer_ratio=0.20,    # 20% exporters
    consumer_ratio=0.60,    # 60% importers
    balanced_ratio=0.20     # 20% flexible
)
```

**Use for:** High prices, grid constraint analysis, peak shaving studies

### Scenario 4: Custom Location

```python
summary = generate_p2p_dataset(
    n_households=30,
    n_days=30,
    interval_minutes=15,
    location=(19.0, 72.8)   # Mumbai coordinates
)
```

---

## Mathematical Details

### Load Profile Generation

#### 1. Base Normalized Profile

The load profile reflects Indian residential consumption patterns with morning (7-9 AM) and evening (8-11 PM) peaks.

Base profile captures:

- **Night hours (00:00-06:00):** 0.25-0.32 p.u. (base fan/fridge load)
- **Morning peak (07:00-09:00):** 0.58-0.65 p.u. (cooking, water heating)
- **Afternoon dip (14:00-15:00):** 0.43-0.46 p.u. (lower activity)
- **Evening peak (20:00-21:00):** 0.95-1.00 p.u. (max: lighting, AC, TV, cooking)

#### 2. Daily Energy Scaling

Given monthly consumption target E_month:

    E_daily = E_month / 30

Daily load profile is scaled:

    L(t) = h(t) × E_daily × M_day

where:

- L(t) = load at hour t (kW)
- h(t) = normalized hourly shape (sum to 1.0)
- E_daily = daily energy target (kWh)
- M_day = random daily multiplier ~ U(0.8, 1.2)

#### 3. Weekday/Weekend Variation

Weekend profiles show elevated daytime (10 AM-5 PM) by 15% and shifted morning peak:

    if 7 <= t < 9:      h_weekend(t) = 0.9 × h_weekday(t)
    if 9 <= t < 11:     h_weekend(t) = 1.1 × h_weekday(t)
    if 10 <= t < 17:    h_weekend(t) = 1.15 × h_weekday(t)
    else:               h_weekend(t) = h_weekday(t)

Then re-normalized: h_weekend = h_weekend / Σ h_weekend

#### 4. Sub-Hourly Interpolation

For intervals < 60 minutes, linear interpolation between hourly values:

    L(t + Δt) = L(t) + (Δt/60) × [L(t+1) - L(t)]

#### 5. Power Outages

Random outages with 15% daily probability and 1-6 hour duration:

    P(outage) = 0.15 per day
    Duration ~ Categorical([1, 2, 3, 4, 6] hours)

During outage periods: L(t) = 0

### PVLib Solar Generation

PVLib uses **PVGIS satellite-derived data** (2005-2020) to model rooftop PV systems.

#### 1. PVGIS Data Source

- **Database:** Satellite-measured solar irradiance (Meteosat imagery)
- **Resolution:** ~5 km grid, hourly values
- **Data quality:** Validated ±5-10% accuracy vs ground stations
- **Coverage:** Full India, worldwide

#### 2. Irradiance Components Retrieved

PVLib extracts three irradiance types:

    GHI(t) = Global Horizontal Irradiance (W/m²)
    DNI(t) = Direct Normal Irradiance (W/m²)
    DHI(t) = Diffuse Horizontal Irradiance (W/m²)

#### 3. Plane-of-Array (POA) Irradiance

Adjusted for tilt angle β = 20° and azimuth γ = 180° (South-facing):

    POA(t) = DNI(t) × cos(θ_z) + DHI(t) × (1+cos(β))/2
             + GHI(t) × ρ × (1-cos(β))/2

where:

- θ_z = solar zenith angle
- ρ = ground reflectance (0.25)
- β = tilt angle (20° for Delhi)

#### 4. Temperature and Efficiency

PV efficiency decreases with module temperature:

    η(T) = η_ref × [1 - γ × (T - T_ref)]

where:

- γ = temperature coefficient (-0.004 per °C typical)
- T_ref = 25°C reference
- Module temperature T ≈ T_ambient + 0.02 × POA

#### 5. System Losses

Total efficiency accounts for multiple loss mechanisms:

    η_total = η_module × η_inverter × η_wiring × η_soiling

Default total loss: 14% (86% final efficiency)

Losses breakdown:

- Module: ~80% (temperature derating)
- Inverter: ~97%
- Wiring/connections: ~98%
- Soiling/dust: ~85%
- Combined: ~86%

#### 6. AC Power Output

Final AC power output at time t:

    P_AC(t) = P_DC_rated × (POA(t) / 1000) × η(T) × η_losses

where:

- P_DC_rated = system capacity (kW)
- POA normalized to 1000 W/m² (Standard Test Conditions)
- Clipped at inverter capacity

#### 7. Time Series Processing

**Hourly → Sub-hourly conversion:**

For interval Δτ minutes (15, 30):

    P(t + Δτ) = P(h) + (Δτ/60) × [P(h+1) - P(h)]

Linear interpolation assumes gradual power ramps (realistic for variable clouds).

---

## Output Format

### CSV Files

Each household gets one CSV file: `HH001.csv`, `HH002.csv`, etc.

**Columns:**

```
timestamp,household_id,load_kW,solar_kW
2024-06-01 00:00:00,HH001,0.3542,0.0000
2024-06-01 01:00:00,HH001,0.3128,0.0000
2024-06-01 12:00:00,HH001,0.4891,1.2534
```

**Data types:**

- `timestamp`: ISO 8601 format (YYYY-MM-DD HH:MM:SS)
- `household_id`: String (HH001, HH002, ...)
- `load_kW`: Float, ≥0 (residential consumption)
- `solar_kW`: Float, ≥0 (rooftop PV generation)

### Summary JSON

`dataset_summary.json` contains:

```json
{
  "configuration": {
    "n_households": 10,
    "n_days": 7,
    "interval_minutes": 60,
    "location": "Delhi",
    "prosumer_ratio": 0.33,
    "consumer_ratio": 0.33,
    "balanced_ratio": 0.34
  },
  "households": [
    {
      "household_id": "HH001",
      "type": "prosumer",
      "monthly_kwh_target": 115.0,
      "pv_kw": 2.93,
      "total_load_kwh": 23.8,
      "total_solar_kwh": 94.2,
      "net_kwh": 70.4,
      "num_outages": 3,
      "records": 168
    }
  ],
  "net_summary": {
    "exporters": 3,
    "importers": 3,
    "total_surplus": 210.5,
    "total_deficit": 180.3
  }
}
```

### Analysis Example

```python
import pandas as pd
import json

# Load household data
df = pd.read_csv('p2p_simulation_data/HH001.csv', parse_dates=['timestamp'])

# Calculate net demand (positive = import, negative = export)
df['net_kW'] = df['load_kW'] - df['solar_kW']

# Find export periods
exports = df[df['net_kW'] < 0]
total_export = -exports['net_kW'].sum() * (1/24)  # kWh

# Load market summary
with open('p2p_simulation_data/dataset_summary.json') as f:
    summary = json.load(f)

print(f"Exporters: {summary['net_summary']['exporters']}")
print(f"Trading potential: {summary['net_summary']['total_surplus']:.1f} kWh")
```

---

## Troubleshooting

### Error: "Ratios must sum to 1.0"

**Cause:** `prosumer_ratio + consumer_ratio + balanced_ratio ≠ 1.0`

**Fix:**

```python
# Check: must equal 1.0
assert abs(0.33 + 0.33 + 0.34 - 1.0) < 0.01
```

### Error: "interval_minutes must be 15, 30, or 60"

**Cause:** Invalid time resolution specified

**Fix:** Use only 15, 30, or 60 minutes

### Warning: "Using synthetic solar profile"

**Cause:** PVGIS API unavailable (no internet or timeout)

**Effect:** Falls back to physics-based synthetic model

- Still produces realistic results
- Based on Delhi sun path and atmospheric effects
- Good enough for algorithm testing

### Large File Sizes

**Problem:** Data is 50+ MB for 100 households × 365 days

**Solutions:**

- Use `interval_minutes=60` (hourly) instead of 15-min
- Generate shorter periods (7-30 days) for testing
- Or compress with: `df.to_csv(..., compression='gzip')`

### All Households Have Surplus

**Cause:** Summer season (June) with heavy solar deployment

**Solutions:**

- Generate winter data (adjust PVGIS year handling)
- Reduce `prosumer_ratio` (more consumers)
- Increase consumption: `monthly_kwh = np.random.uniform(250, 350)`

---

## Next Steps: P2P Algorithm Implementation

1. **Load all households:**

   ```python
   import glob
   households = [pd.read_csv(f) for f in glob.glob('p2p_simulation_data/*.csv')]
   ```

2. **Identify surplus/deficit per time interval:**

   ```python
   for hh in households:
       hh['net_kW'] = hh['load_kW'] - hh['solar_kW']
   ```

3. **Implement P2P matching:**

   - Auction-based: Surplus bids price, deficit buys
   - Bilateral: Fixed price negotiation
   - Cooperative: Maximize community self-consumption

4. **Calculate benefits:**

   - Cost savings per household
   - Grid load reduction
   - Peak demand shaving

5. **Sensitivity analysis:**
   Run with different ratios to understand market dynamics

---

## References

- PVGIS: https://joint-research-centre.ec.europa.eu/pvgis/
- pvlib Documentation: https://pvlib-python.readthedocs.io/
- Indian Load Profiles: iAWE Dataset, I-BLEND Dataset
- P2P Energy Trading: Tushar et al. (2020), IEEE Reviews

---

**Last Updated:** November 1, 2025
