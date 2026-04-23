"""
Food Delivery Dispatching -- Queueing Theory (G/G/c)
=====================================================
Computes lambda, mu, c, rho, capacity_j, queue_urgency_i
using cleaned data from the Datasets folder.

Outputs (written to same folder as this script):
  zone_date_queueing.csv   -- zone x date level queueing stats
  courier_capacity.csv     -- per-courier capacity at each checkpoint
  orders_with_urgency.csv  -- per-order queue urgency
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "Datasets"
OUTPUT_DIR   = Path(__file__).parent
MAX_CAPACITY = 10
URGENCY_CAP_SEC = 7200   # cap urgency at 2 hours to suppress outliers

# dispatch checkpoints span ~5 minutes per day (03:25 -- 03:30)
WINDOW_SEC = 300.0


def zone_from_coords(lat, lng, grid_size=0.01):
    lat_bin = (lat // grid_size).astype(int)
    lng_bin = (lng // grid_size).astype(int)
    return lat_bin.astype(str) + "_" + lng_bin.astype(str)


# =============================================================================
# 1. LOAD DATA
# Coordinates in the cleaned CSVs are already in decimal degrees -- no scaling.
# Time columns are stored as strings; parse explicitly.
# =============================================================================
print("Loading data...")

all_info = pd.read_csv(DATASETS_DIR / "all_info.csv")
waybill  = pd.read_csv(DATASETS_DIR / "waybill.csv")
rider    = pd.read_csv(DATASETS_DIR / "rider.csv")
courier  = pd.read_csv(DATASETS_DIR / "courier.csv")

TIME_COLS_AI = [
    "dispatch_time", "grab_time", "fetch_time", "arrive_time",
    "estimate_arrived_time", "estimate_meal_prepare_time", "platform_order_time",
]
for col in TIME_COLS_AI:
    if col in all_info.columns:
        all_info[col] = pd.to_datetime(all_info[col], errors="coerce")

waybill["dispatch_time"]     = pd.to_datetime(waybill["dispatch_time"],     errors="coerce")
rider["dispatch_time"]       = pd.to_datetime(rider["dispatch_time"],       errors="coerce")
courier["wave_start_time"]   = pd.to_datetime(courier["wave_start_time"],   errors="coerce")
courier["wave_end_time"]     = pd.to_datetime(courier["wave_end_time"],     errors="coerce")

EPOCH = pd.Timestamp("1970-01-01")

print(f"  all_info : {len(all_info):,} rows")
print(f"  waybill  : {len(waybill):,} rows")
print(f"  rider    : {len(rider):,} rows")
print(f"  courier  : {len(courier):,} rows")


# =============================================================================
# 2. ZONE ASSIGNMENT
# =============================================================================
# -- pending orders: join waybill (checkpoint) with all_info for coordinates
waybill_geo = waybill.merge(
    all_info[["order_id", "sender_lat", "sender_lng", "platform_order_time"]]
    .drop_duplicates("order_id"),
    on="order_id", how="left",
)
waybill_geo = waybill_geo.dropna(subset=["sender_lat"])
waybill_geo["zone_id"] = zone_from_coords(waybill_geo["sender_lat"], waybill_geo["sender_lng"])
waybill_geo["date"]    = waybill_geo["dispatch_time"].dt.date
print(f"\n  Pending orders with coordinates: {len(waybill_geo):,}")

# -- couriers: zone from rider snapshot location
rider["zone_id"] = zone_from_coords(rider["rider_lat"], rider["rider_lng"])
rider["date"]    = rider["dispatch_time"].dt.date


# =============================================================================
# 3. ESTIMATE LAMBDA -- ORDER ARRIVAL RATE PER ZONE-DATE
# All checkpoints fall within a 5-minute window per day, so WINDOW_SEC = 300.
# =============================================================================
print("\nEstimating lambda (arrival rate) per zone-date...")

lambda_df = (
    waybill_geo
    .groupby(["zone_id", "date"])
    .agg(order_count=("order_id", "count"))
    .reset_index()
)
lambda_df["lambda_per_sec"] = lambda_df["order_count"] / WINDOW_SEC
lambda_df["lambda_per_min"] = lambda_df["lambda_per_sec"] * 60

print(f"  Zone-dates computed : {len(lambda_df):,}")
print(f"  Mean lambda         : {lambda_df['lambda_per_min'].mean():.4f} orders/min")
print(f"  Max  lambda         : {lambda_df['lambda_per_min'].max():.4f} orders/min (peak zone)")


# =============================================================================
# 4. ESTIMATE MU -- COURIER SERVICE RATE
# Service time = grab_time -> arrive_time.
# This is the time the courier is actively occupied per order, which is the
# correct service time for G/G/c. We use the same definition for W (Little's
# Law) to keep mu and W consistent.
# =============================================================================
print("\nEstimating mu (service rate)...")

grabbed = all_info[all_info["is_courier_grabbed"] == 1].copy()
grabbed = grabbed[
    (grabbed["grab_time"] > EPOCH) &
    (grabbed["arrive_time"] > EPOCH)
]
grabbed["service_sec"] = (grabbed["arrive_time"] - grabbed["grab_time"]).dt.total_seconds()
grabbed = grabbed[grabbed["service_sec"].between(60, 5400)]   # 1 min -- 90 min

courier_mu = (
    grabbed
    .groupby("courier_id")["service_sec"]
    .agg(avg_service_sec="mean", n_deliveries="count")
    .reset_index()
)
courier_mu = courier_mu[courier_mu["n_deliveries"] >= 5]
courier_mu["mu_per_sec"] = 1.0 / courier_mu["avg_service_sec"]

global_service_sec = grabbed["service_sec"].median()
global_mu_per_sec  = 1.0 / global_service_sec
global_mu_per_min  = global_mu_per_sec * 60

print(f"  Couriers with mu estimate  : {len(courier_mu):,}")
print(f"  Global median service time : {global_service_sec:.0f}s ({global_service_sec/60:.1f} min)")
print(f"  Global mu                  : {global_mu_per_min:.5f} deliveries/min")


# =============================================================================
# 5. COMPUTE c -- UNIQUE ACTIVE COURIERS PER ZONE-DATE
# =============================================================================
print("\nComputing c (active couriers) per zone-date...")

courier_count = (
    rider
    .groupby(["zone_id", "date"])["courier_id"]
    .nunique()
    .reset_index()
    .rename(columns={"courier_id": "c_couriers"})
)
print(f"  Zone-dates with courier data : {len(courier_count):,}")
print(f"  Mean couriers per zone-date  : {courier_count['c_couriers'].mean():.1f}")


# =============================================================================
# 6. COMPUTE RHO -- SYSTEM UTILIZATION  rho = lambda / (c * mu)
# G/G/c requires rho < 1 for a stable queue. Zones where rho >= 1 indicate
# demand exceeds supply; we clip to 0.99 and flag them.
# =============================================================================
print("\nComputing rho (system utilization)...")

ggc = lambda_df.merge(courier_count, on=["zone_id", "date"], how="left")
ggc["c_couriers"] = ggc["c_couriers"].fillna(1).astype(int)
ggc["mu_per_sec"] = global_mu_per_sec
ggc["rho_raw"]    = ggc["lambda_per_sec"] / (ggc["c_couriers"] * ggc["mu_per_sec"])
ggc["overloaded"] = ggc["rho_raw"] >= 1
ggc["rho"]        = ggc["rho_raw"].clip(upper=0.99)   # enforce stability for G/G/c

n_over = ggc["overloaded"].sum()
print(f"  Overloaded zone-dates (rho_raw >= 1) : {n_over} ({n_over/len(ggc):.1%})")
print(f"  Mean rho (raw)  : {ggc['rho_raw'].mean():.3f}")
print(f"  Mean rho (clip) : {ggc['rho'].mean():.3f}")


# =============================================================================
# 7. DYNAMIC COURIER CAPACITY via LITTLE'S LAW  L = lambda * W
# W = median service time (grab->arrive), consistent with mu.
# L_per_courier = (lambda / c) * W  gives the average number of orders a
# courier is actively handling. This becomes capacity_j.
# =============================================================================
print("\nComputing dynamic capacity_j via Little's Law...")

grabbed["zone_id"] = zone_from_coords(grabbed["sender_lat"], grabbed["sender_lng"])
grabbed["date"]    = grabbed["dispatch_time"].dt.date

W_zone_date = (
    grabbed
    .groupby(["zone_id", "date"])["service_sec"]
    .median()
    .reset_index()
    .rename(columns={"service_sec": "W_sec"})
)

ggc = ggc.merge(W_zone_date, on=["zone_id", "date"], how="left")
ggc["W_sec"] = ggc["W_sec"].fillna(global_service_sec)

ggc["lambda_per_courier_sec"] = ggc["lambda_per_sec"] / ggc["c_couriers"].clip(lower=1)
ggc["L_per_courier"]          = ggc["lambda_per_courier_sec"] * ggc["W_sec"]
ggc["capacity"] = ggc["L_per_courier"].clip(lower=1, upper=MAX_CAPACITY).round().astype(int)

print(f"  Mean capacity per courier : {ggc['capacity'].mean():.2f}")
print(f"  Capacity distribution:")
print(ggc["capacity"].value_counts().sort_index().to_string())


# =============================================================================
# 8. ATTACH CAPACITY TO INDIVIDUAL COURIERS
# =============================================================================
print("\nAttaching capacity_j to individual couriers...")

capacity_lookup = ggc[["zone_id", "date", "capacity"]].copy()

courier_capacity = (
    rider[["courier_id", "dispatch_time", "zone_id", "date"]]
    .drop_duplicates(subset=["courier_id", "dispatch_time"])
)
courier_capacity = courier_capacity.merge(capacity_lookup, on=["zone_id", "date"], how="left")
courier_capacity["capacity"] = courier_capacity["capacity"].fillna(3).astype(int)

courier_capacity = courier_capacity.merge(
    courier_mu[["courier_id", "mu_per_sec", "avg_service_sec"]],
    on="courier_id", how="left",
)
courier_capacity["mu_per_sec"]      = courier_capacity["mu_per_sec"].fillna(global_mu_per_sec)
courier_capacity["avg_service_sec"] = courier_capacity["avg_service_sec"].fillna(global_service_sec)
print(f"  Courier-dispatch snapshots : {len(courier_capacity):,}")


# =============================================================================
# 9. QUEUE URGENCY  urgency_i = checkpoint_time - platform_order_time
# Represents how long the customer has been waiting since placing the order.
# Capped at URGENCY_CAP_SEC to suppress data errors (e.g. orders placed days
# before the dispatch window).
# =============================================================================
print("\nComputing queue urgency per order...")

urgency = waybill_geo.copy()
urgency = urgency[
    urgency["platform_order_time"].notna() &
    (urgency["platform_order_time"] > EPOCH)
]
urgency["queue_urgency_sec"] = (
    urgency["dispatch_time"] - urgency["platform_order_time"]
).dt.total_seconds().clip(lower=0, upper=URGENCY_CAP_SEC)
urgency["queue_urgency_min"] = urgency["queue_urgency_sec"] / 60

print(f"  Orders with urgency  : {len(urgency):,}")
print(f"  Mean wait time       : {urgency['queue_urgency_min'].mean():.2f} min")
print(f"  Median wait          : {urgency['queue_urgency_min'].median():.2f} min")
print(f"  90th pct wait        : {urgency['queue_urgency_min'].quantile(0.9):.2f} min")
print(f"  % waiting > 5 min   : {(urgency['queue_urgency_min'] > 5).mean():.1%}")


# =============================================================================
# 10. SAVE OUTPUTS
# =============================================================================
print("\nSaving outputs...")

ggc.to_csv(OUTPUT_DIR / "zone_date_queueing.csv", index=False)
print(f"  zone_date_queueing.csv   ({len(ggc):,} zone-dates)")

courier_capacity.to_csv(OUTPUT_DIR / "courier_capacity.csv", index=False)
print(f"  courier_capacity.csv     ({len(courier_capacity):,} courier snapshots)")

orders_out = urgency[[
    "order_id", "dispatch_time", "zone_id", "date",
    "queue_urgency_sec", "queue_urgency_min",
]].copy()
orders_out.to_csv(OUTPUT_DIR / "orders_with_urgency.csv", index=False)
print(f"  orders_with_urgency.csv  ({len(orders_out):,} orders)")


# =============================================================================
# 11. SUMMARY
# =============================================================================
print("\n" + "=" * 55)
print("QUEUEING THEORY SUMMARY")
print("=" * 55)
print(ggc[["lambda_per_min", "c_couriers", "rho", "W_sec", "capacity"]].describe().round(3).to_string())
print("\nDone.")
