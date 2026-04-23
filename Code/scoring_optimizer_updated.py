"""
Food Delivery Dispatching -- Scoring + Optimizer
=================================================
Uses LightGBM ETA and acceptance-probability models, queueing theory outputs,
and a logistic-regression weight calibration step.

Score(i, j) = -alpha * ETA_ij
            + beta  * P_accept(i, j)
            - gamma * d_pickup_ij
            - delta * courier_load_j          <- note: NEGATIVE sign penalizes
            + theta * queue_urgency_i         <- busy couriers, contrary to
                                                 the +delta in the proposal
                                                 formula; this code is correct.

Run queueing_theory.py first to generate:
  zone_date_queueing.csv
  courier_capacity.csv
  orders_with_urgency.csv
"""

import ast
import warnings
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from scipy.optimize import linear_sum_assignment
from pathlib import Path

warnings.filterwarnings("ignore")

DATASETS_DIR = Path(__file__).parent.parent / "Datasets"
MODELS_DIR   = Path(__file__).parent.parent / "ML Models"
QUEUE_DIR    = Path(__file__).parent          # outputs from queueing_theory.py

EPOCH        = pd.Timestamp("1970-01-01")
URGENCY_CAP  = 7200


# =============================================================================
# HELPERS
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def angle_diff_norm(lat_r, lng_r, lat_p, lng_p, lat_d, lng_d):
    """
    Angle between rider->pickup and pickup->delivery vectors, normalised to [0,1].
    0 = perfectly aligned (no detour), 1 = complete U-turn.
    Uses flat-earth approximation; fine for small city-scale distances.
    """
    v1_lat, v1_lng = lat_p - lat_r, lng_p - lng_r
    v2_lat, v2_lng = lat_d - lat_p, lng_d - lng_p
    norm1 = np.sqrt(v1_lat ** 2 + v1_lng ** 2) + 1e-9
    norm2 = np.sqrt(v2_lat ** 2 + v2_lng ** 2) + 1e-9
    cos_a = (v1_lat * v2_lat + v1_lng * v2_lng) / (norm1 * norm2)
    return float(np.arccos(np.clip(cos_a, -1.0, 1.0)) / np.pi)


def parse_load(x):
    try:
        return len(ast.literal_eval(x)) if isinstance(x, str) else 0
    except Exception:
        return 0


def zone_from_coords(lat, lng, grid_size=0.01):
    return f"{int(lat // grid_size)}_{int(lng // grid_size)}"


# =============================================================================
# 1. LOAD ML MODELS
# =============================================================================
print("Loading ML models...")

eta_model    = lgb.Booster(model_file=str(MODELS_DIR / "eta_model_optimized.txt"))
accept_model = lgb.Booster(model_file=str(MODELS_DIR / "accept_model.txt"))

with open(MODELS_DIR / "eta_model_meta.pkl",    "rb") as f:
    eta_meta = pickle.load(f)
with open(MODELS_DIR / "accept_model_meta.pkl", "rb") as f:
    accept_meta = pickle.load(f)

FEATURES_ETA    = eta_meta["features_eta"]
FEATURES_ACCEPT = accept_meta["features_accept"]
print(f"  ETA    features ({len(FEATURES_ETA)}): {FEATURES_ETA}")
print(f"  Accept features ({len(FEATURES_ACCEPT)}): {FEATURES_ACCEPT}")


# =============================================================================
# 2. LOAD RAW DATA
# Coordinates are already in decimal degrees; no scaling required.
# All time columns are strings -- parse explicitly.
# =============================================================================
print("\nLoading data...")

all_info    = pd.read_csv(DATASETS_DIR / "all_info.csv")
rider_raw   = pd.read_csv(DATASETS_DIR / "rider.csv")
waybill_raw = pd.read_csv(DATASETS_DIR / "waybill.csv")
wave        = pd.read_csv(DATASETS_DIR / "courier.csv")

for col in ["dispatch_time", "grab_time", "arrive_time",
            "estimate_meal_prepare_time", "platform_order_time"]:
    if col in all_info.columns:
        all_info[col] = pd.to_datetime(all_info[col], errors="coerce")

rider_raw["dispatch_time"]   = pd.to_datetime(rider_raw["dispatch_time"],   errors="coerce")
waybill_raw["dispatch_time"] = pd.to_datetime(waybill_raw["dispatch_time"], errors="coerce")
wave["wave_start_time"]      = pd.to_datetime(wave["wave_start_time"],      errors="coerce")
wave["wave_end_time"]        = pd.to_datetime(wave["wave_end_time"],        errors="coerce")


# =============================================================================
# 3. LOAD QUEUEING OUTPUTS
# =============================================================================
print("Loading queueing outputs...")
courier_capacity = pd.read_csv(QUEUE_DIR / "courier_capacity.csv",
                               parse_dates=["dispatch_time"])
orders_urgency   = pd.read_csv(QUEUE_DIR / "orders_with_urgency.csv",
                               parse_dates=["dispatch_time"])
ggc_df           = pd.read_csv(QUEUE_DIR / "zone_date_queueing.csv",
                               parse_dates=["date"])

print(f"  Courier capacity snapshots : {len(courier_capacity):,}")
print(f"  Orders with urgency        : {len(orders_urgency):,}")


# =============================================================================
# 4. BUILD HISTORICAL FEATURES FOR CALIBRATION AND LOOKUP
# =============================================================================
print("\nBuilding historical feature lookups...")

grabbed_hist = all_info[
    (all_info["is_courier_grabbed"] == 1) &
    (all_info["grab_time"]  > EPOCH) &
    (all_info["arrive_time"] > EPOCH)
].copy()
grabbed_hist["eta_sec"] = (
    grabbed_hist["arrive_time"] - grabbed_hist["dispatch_time"]
).dt.total_seconds()
grabbed_hist = grabbed_hist[grabbed_hist["eta_sec"].between(60, 7200)]

# poi_avg_eta: historical average delivery time per restaurant
poi_avg_eta = (
    grabbed_hist.groupby("poi_id")["eta_sec"].mean()
    .reset_index().rename(columns={"eta_sec": "poi_avg_eta"})
)
global_avg_eta = grabbed_hist["eta_sec"].median()

# rider_avg_eta: historical average delivery time per courier
rider_avg_eta = (
    grabbed_hist.groupby("courier_id")["eta_sec"].mean()
    .reset_index().rename(columns={"eta_sec": "rider_avg_eta"})
)

# hist_poi_wait: average meal prep wait per restaurant
grabbed_hist["wait_meal_sec"] = (
    (grabbed_hist["estimate_meal_prepare_time"] - grabbed_hist["dispatch_time"])
    .dt.total_seconds().clip(lower=0)
)
hist_poi_wait = (
    grabbed_hist.groupby("poi_id")["wait_meal_sec"].mean()
    .reset_index().rename(columns={"wait_meal_sec": "hist_poi_wait"})
)

# hist courier features from wave data
wave["wave_duration_sec"] = (wave["wave_end_time"] - wave["wave_start_time"]).dt.total_seconds()
hist_wave = (
    wave.groupby("courier_id")
    .agg(hist_avg_wave_dur=("wave_duration_sec", "mean"),
         hist_avg_wave_cnt=("wave_id",           "count"))
    .reset_index()
)

hist_grab_rate = (
    all_info.groupby("courier_id")["is_courier_grabbed"].mean()
    .reset_index().rename(columns={"is_courier_grabbed": "hist_grab_rate"})
)

# supply_demand_ratio per zone-date: c_couriers / order_count
ggc_df["supply_demand_ratio"] = (
    ggc_df["c_couriers"] / ggc_df["order_count"].replace(0, np.nan)
).fillna(1.0)


# =============================================================================
# 5. WEIGHT CALIBRATION via LOGISTIC REGRESSION
# Positive examples: actual (order, courier) matched pairs from all_info.
# Negative examples: same orders paired with randomly shuffled couriers
#                    within the same date.
#
# ETA is an order-level feature -- identical for pos and neg examples of the
# same order -- so logistic regression cannot calibrate alpha from it. Instead
# alpha is set by empirical normalisation (1 / eta_std) so that a one-std
# change in ETA contributes one unit to the score.
# Courier-level features (d_pickup, hist_grab_rate / P_accept proxy, urgency)
# DO differ across positive vs negative pairs and are calibrated by LR.
# =============================================================================
print("\nCalibrating weights via logistic regression...")

def calibrate_weights(all_info_df, eta_std):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        pos = all_info_df[
            (all_info_df["is_courier_grabbed"] == 1) &
            (all_info_df["grab_time"]  > EPOCH) &
            (all_info_df["grab_lat"]   != 0) &
            (all_info_df["grab_lng"]   != 0) &
            (all_info_df["arrive_time"] > EPOCH)
        ].copy()

        pos["d_pickup_km"] = haversine_km(
            pos["grab_lat"].values, pos["grab_lng"].values,
            pos["sender_lat"].values, pos["sender_lng"].values,
        )
        pos = pos[pos["d_pickup_km"] < 15]

        pos["urgency_sec"] = (
            pos["dispatch_time"] - pos["platform_order_time"]
        ).dt.total_seconds().clip(0, URGENCY_CAP).fillna(0)

        grab_rate_map = all_info_df.groupby("courier_id")["is_courier_grabbed"].mean()
        pos["hist_grab_rate"] = pos["courier_id"].map(grab_rate_map).fillna(0.5)

        pos["date"] = pos["dispatch_time"].dt.date

        # Negative: shuffle courier_id within each date so d_pickup and
        # hist_grab_rate differ, while urgency stays the same (order-level).
        np.random.seed(42)
        neg = pos.copy()
        neg["courier_id_neg"] = neg.groupby("date")["courier_id"].transform(
            lambda x: np.random.permutation(x.values)
        )
        courier_locs = (
            pos[["courier_id", "grab_lat", "grab_lng"]]
            .drop_duplicates("courier_id")
            .rename(columns={"courier_id": "courier_id_neg",
                             "grab_lat": "grab_lat_neg",
                             "grab_lng": "grab_lng_neg"})
        )
        neg = neg.merge(courier_locs, on="courier_id_neg", how="left")
        neg["grab_lat_neg"] = neg["grab_lat_neg"].fillna(neg["grab_lat"])
        neg["grab_lng_neg"] = neg["grab_lng_neg"].fillna(neg["grab_lng"])
        neg["d_pickup_km"]   = haversine_km(
            neg["grab_lat_neg"].values, neg["grab_lng_neg"].values,
            neg["sender_lat"].values,   neg["sender_lng"].values,
        )
        neg["hist_grab_rate"] = neg["courier_id_neg"].map(grab_rate_map).fillna(0.5)

        # ETA intentionally excluded: same value for pos and neg of the same order
        FEAT = ["d_pickup_km", "hist_grab_rate", "urgency_sec"]
        pos_feat = pos[FEAT].copy(); pos_feat["label"] = 1
        neg_feat = neg[FEAT].copy(); neg_feat["label"] = 0

        df = pd.concat([pos_feat, neg_feat], ignore_index=True).dropna()
        n  = min(50_000, (df["label"] == 1).sum(), (df["label"] == 0).sum())
        df_bal = pd.concat([
            df[df["label"] == 1].sample(n, random_state=42),
            df[df["label"] == 0].sample(n, random_state=42),
        ], ignore_index=True)

        X = df_bal[FEAT].values
        y = df_bal["label"].values
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        lr     = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_sc, y)

        # raw-feature coefficients: [d_pickup, p_accept_proxy, urgency]
        raw = lr.coef_[0] / scaler.scale_
        print(f"  LR coefs (raw space): d_pickup={raw[0]:.2e}, "
              f"p_accept={raw[1]:.2e}, urgency={raw[2]:.2e}")

        # gamma must penalise, beta must reward, theta must reward
        gamma = max(0.0, -raw[0])
        beta  = max(0.0,  raw[1])
        theta = max(0.0,  raw[2])

        # alpha is set by empirical normalisation so 1-std ETA ~ 1 score unit
        alpha = 1.0 / max(eta_std, 1.0)

        # Normalise so the dominant penalty (gamma or alpha) equals 1.0
        dominant = max(alpha, gamma, 1e-9)
        weights = {
            "alpha": alpha  / dominant,
            "beta":  max(beta  / dominant, 0.3),   # floor for P_accept
            "gamma": gamma  / dominant,
            "delta": 0.5,                           # mild load penalty (fixed)
            "theta": theta  / dominant,
        }
        return weights

    except ImportError:
        print("  sklearn not available; using normalised empirical weights")
        return None


eta_std = grabbed_hist["eta_sec"].std()
weights = calibrate_weights(all_info, eta_std)

if weights is None:
    # Fallback: scale by empirical feature stds so units are comparable
    d_std   = haversine_km(
        grabbed_hist["grab_lat"].values, grabbed_hist["grab_lng"].values,
        grabbed_hist["sender_lat"].values, grabbed_hist["sender_lng"].values,
    ).std() if len(grabbed_hist) > 0 else 1.0
    urg_std = orders_urgency["queue_urgency_sec"].std() if len(orders_urgency) > 0 else 300.0

    weights = {
        "alpha": 1.0 / max(eta_std, 1),
        "beta":  1.5,
        "gamma": 1.0 / max(d_std, 0.1),
        "delta": 0.5,
        "theta": 1.0 / max(urg_std, 1),
    }

print(f"  Calibrated weights: { {k: round(v, 6) for k, v in weights.items()} }")
pd.DataFrame([{"weight": k, "value": v} for k, v in weights.items()]).to_csv(
    QUEUE_DIR / "weight_calibration.csv", index=False
)


# =============================================================================
# 6. BUILD DISPATCH BATCH FOR DEMO
# =============================================================================
print("\nPreparing dispatch batch...")

TEST_DATE_STR = "2022-10-19"

# -- Pending orders at first checkpoint of the test date
pending_raw = waybill_raw[
    waybill_raw["dispatch_time"].dt.strftime("%Y-%m-%d") == TEST_DATE_STR
].copy()
first_checkpoint = pending_raw["dispatch_time"].min()
pending_snap = pending_raw[pending_raw["dispatch_time"] == first_checkpoint].copy()

# Join with all_info for spatial + meal features
BASE_COLS = ["order_id", "sender_lat", "sender_lng", "recipient_lat", "recipient_lng",
             "estimate_meal_prepare_time", "is_prebook", "poi_id", "platform_order_time"]
pending_snap = pending_snap.merge(
    all_info[BASE_COLS].drop_duplicates("order_id"),
    on="order_id", how="left",
)
pending_snap = pending_snap.dropna(subset=["sender_lat"])

# Derived order features
pending_snap["dispatch_dt"]    = pending_snap["dispatch_time"]
pending_snap["hour_of_day"]    = pending_snap["dispatch_dt"].dt.hour
pending_snap["day_of_week"]    = pending_snap["dispatch_dt"].dt.dayofweek
pending_snap["is_weekend"]     = (pending_snap["day_of_week"] >= 5).astype(int)
pending_snap["is_lunch_peak"]  = pending_snap["hour_of_day"].between(11, 13).astype(int)
pending_snap["is_dinner_peak"] = pending_snap["hour_of_day"].between(17, 20).astype(int)
pending_snap["wait_meal_sec"]  = (
    pending_snap["estimate_meal_prepare_time"] - pending_snap["dispatch_time"]
).dt.total_seconds().clip(lower=0).fillna(0)

# local_demand: orders per poi in this batch snapshot
pending_snap["local_demand"] = (
    pending_snap.groupby("poi_id")["order_id"].transform("count")
)

# poi-level lookups
pending_snap = pending_snap.merge(poi_avg_eta,  on="poi_id", how="left")
pending_snap = pending_snap.merge(hist_poi_wait, on="poi_id", how="left")
pending_snap["poi_avg_eta"]  = pending_snap["poi_avg_eta"].fillna(global_avg_eta)
pending_snap["hist_poi_wait"] = pending_snap["hist_poi_wait"].fillna(0)

# Queue urgency
pending_snap = pending_snap.merge(
    orders_urgency[["order_id", "queue_urgency_sec"]], on="order_id", how="left"
)
pending_snap["queue_urgency_sec"] = pending_snap["queue_urgency_sec"].fillna(0)

# zone for supply_demand_ratio lookup
test_date = pd.to_datetime(TEST_DATE_STR).date()
pending_snap["zone_id"] = pending_snap.apply(
    lambda r: zone_from_coords(r["sender_lat"], r["sender_lng"]), axis=1
)
zone_sd = (
    ggc_df[ggc_df["date"].dt.date == test_date][["zone_id", "supply_demand_ratio"]]
)
pending_snap = pending_snap.merge(zone_sd, on="zone_id", how="left")
pending_snap["supply_demand_ratio"] = pending_snap["supply_demand_ratio"].fillna(1.0)

# -- Candidate couriers at first checkpoint
cand_raw = rider_raw[
    rider_raw["dispatch_time"] == first_checkpoint
].copy()

# Coordinates already in decimal degrees
cand_raw["courier_load"] = cand_raw["current_load"]

# Attach queueing capacity
cand_raw = cand_raw.merge(
    courier_capacity[["courier_id", "dispatch_time", "capacity"]].drop_duplicates(),
    on=["courier_id", "dispatch_time"], how="left",
)
cand_raw["capacity"] = cand_raw["capacity"].fillna(3).astype(int)

# Historical courier features
cand_raw = cand_raw.merge(hist_wave,      on="courier_id", how="left")
cand_raw = cand_raw.merge(hist_grab_rate, on="courier_id", how="left")
cand_raw = cand_raw.merge(rider_avg_eta,  on="courier_id", how="left")
for col, fill in [("hist_avg_wave_dur", cand_raw.get("hist_avg_wave_dur", pd.Series()).median()),
                  ("hist_avg_wave_cnt", cand_raw.get("hist_avg_wave_cnt", pd.Series()).median()),
                  ("hist_grab_rate",    0.5),
                  ("rider_avg_eta",     global_avg_eta)]:
    if col in cand_raw.columns:
        cand_raw[col] = cand_raw[col].fillna(fill if pd.notna(fill) else 0)

orders   = pending_snap.reset_index(drop=True)
couriers = cand_raw.reset_index(drop=True)
n_orders, n_couriers = len(orders), len(couriers)

print(f"  Checkpoint           : {first_checkpoint}")
print(f"  Pending orders       : {n_orders}")
print(f"  Candidate couriers   : {n_couriers}")
print(f"  Pairs to score       : {n_orders * n_couriers:,}")


# =============================================================================
# 7. BUILD PAIRWISE FEATURE MATRIX  (vectorised -- no Python loops)
# Strategy: extract (n_orders,) and (n_couriers,) arrays, then broadcast to
# (n_orders, n_couriers) arrays for all distance / derived features.
# Final pair matrix is built by flattening in row-major (order-major) order.
# =============================================================================
print("\nBuilding pairwise feature matrix (vectorised)...")

# -- Order arrays (n_orders,)
o_lat  = orders["sender_lat"].values
o_lng  = orders["sender_lng"].values
r_lat  = orders["recipient_lat"].values
r_lng  = orders["recipient_lng"].values

# -- Courier arrays (n_couriers,)
c_lat  = couriers["rider_lat"].values
c_lng  = couriers["rider_lng"].values

# Broadcast all coordinates to (n_orders, n_couriers) by reshaping
# orders as (n_orders, 1) and couriers as (1, n_couriers).
def hav_bc(lat1, lon1, lat2, lon2):
    """Haversine distance in km, supports numpy broadcasting."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# All result shapes: (n_orders, n_couriers)
d_pickup   = hav_bc(c_lat[None, :], c_lng[None, :], o_lat[:, None], o_lng[:, None])
d_delivery = hav_bc(o_lat[:, None], o_lng[:, None], r_lat[:, None], r_lng[:, None])
d_total    = d_pickup + d_delivery
d_direct   = hav_bc(c_lat[None, :], c_lng[None, :], r_lat[:, None], r_lng[:, None])

detour_km    = d_total - d_direct
detour_ratio = detour_km / (d_total + 0.1)

# courier_load broadcast: (1, n_couriers)
load_arr     = couriers["courier_load"].values[None, :]
load_per_km  = load_arr / (d_total + 0.5)

# angle_diff: angle between (courier->pickup) and (pickup->delivery) vectors
# Computed on lat/lng differences (valid for small city-scale areas).
v1_lat = o_lat[:, None] - c_lat[None, :]    # (n_orders, n_couriers)
v1_lng = o_lng[:, None] - c_lng[None, :]
v2_lat = (r_lat - o_lat)[:, None]           # (n_orders, 1) -> broadcasts
v2_lng = (r_lng - o_lng)[:, None]
norm1  = np.sqrt(v1_lat ** 2 + v1_lng ** 2) + 1e-9
norm2  = np.sqrt(v2_lat ** 2 + v2_lng ** 2) + 1e-9
cos_a  = (v1_lat * v2_lat + v1_lng * v2_lng) / (norm1 * norm2)
angle_diff_mat = np.arccos(np.clip(cos_a, -1.0, 1.0)) / np.pi  # (n_orders, n_couriers)

# supply_demand_ratio per courier zone (looked up per courier, broadcast over orders)
cour_zones   = couriers.apply(lambda r: zone_from_coords(r["rider_lat"], r["rider_lng"]), axis=1)
ggc_test     = ggc_df[ggc_df["date"].dt.date == test_date].set_index("zone_id")["supply_demand_ratio"]
sd_arr       = cour_zones.map(ggc_test).fillna(1.0).values[None, :]  # (1, n_couriers)

# Order-level features broadcast: (n_orders, 1)
def obcast(col, default=0):
    return orders[col].fillna(default).values[:, None]

# Courier-level features broadcast: (1, n_couriers)
def cbcast(col, default=0):
    return couriers[col].fillna(default).values[None, :]

# Flatten everything to (n_orders * n_couriers,) in order-major order
N = n_orders * n_couriers

pairs = pd.DataFrame({
    "order_id":             np.repeat(orders["order_id"].values, n_couriers),
    "courier_id":           np.tile(couriers["courier_id"].values, n_orders),
    # distances  (d_delivery is order-level → repeat for each courier)
    "d_delivery_km":        np.repeat(d_delivery.ravel(), n_couriers),
    "d_pickup_km":          d_pickup.ravel(),
    "d_total_km":           d_total.ravel(),
    "d_rider_to_sender_km": d_pickup.ravel(),
    # time (order-level)
    "hour_of_day":          np.repeat(orders["hour_of_day"].values,   n_couriers),
    "day_of_week":          np.repeat(orders["day_of_week"].values,   n_couriers),
    "is_weekend":           np.repeat(orders["is_weekend"].values,    n_couriers),
    "is_lunch_peak":        np.repeat(orders["is_lunch_peak"].values, n_couriers),
    "is_dinner_peak":       np.repeat(orders["is_dinner_peak"].values,n_couriers),
    "wait_meal_sec":        np.repeat(orders["wait_meal_sec"].fillna(0).values, n_couriers),
    "is_prebook":           np.repeat(orders["is_prebook"].fillna(0).values, n_couriers),
    "hist_poi_wait":        np.repeat(orders["hist_poi_wait"].fillna(0).values, n_couriers),
    "local_demand":         np.repeat(orders["local_demand"].fillna(1).values, n_couriers),
    "poi_avg_eta":          np.repeat(orders["poi_avg_eta"].fillna(global_avg_eta).values, n_couriers),
    "queue_urgency_sec":    np.repeat(orders["queue_urgency_sec"].fillna(0).values, n_couriers),
    # courier-level
    "courier_load":         np.tile(couriers["courier_load"].fillna(0).values,   n_orders),
    "rider_lat":            np.tile(couriers["rider_lat"].values,                n_orders),
    "rider_lng":            np.tile(couriers["rider_lng"].values,                n_orders),
    "hist_avg_wave_dur":    np.tile(couriers["hist_avg_wave_dur"].fillna(0).values, n_orders),
    "hist_avg_wave_cnt":    np.tile(couriers["hist_avg_wave_cnt"].fillna(0).values, n_orders),
    "hist_grab_rate":       np.tile(couriers["hist_grab_rate"].fillna(0.5).values,  n_orders),
    "rider_avg_eta":        np.tile(couriers["rider_avg_eta"].fillna(global_avg_eta).values, n_orders),
    "capacity_j":           np.tile(couriers["capacity"].values, n_orders),
    # pair-level derived
    "detour_km":            detour_km.ravel(),
    "detour_ratio":         detour_ratio.ravel(),
    "load_per_km":          load_per_km.ravel(),
    "supply_demand_ratio":  np.tile(cour_zones.map(ggc_test).fillna(1.0).values, n_orders),
    "angle_diff":           angle_diff_mat.ravel(),
})
pairs = pairs.fillna(0)
print(f"  Pair matrix: {pairs.shape}")


# =============================================================================
# 8. ML INFERENCE
# =============================================================================
print("\nRunning ML inference...")

X_eta    = pairs[FEATURES_ETA].fillna(0)
X_accept = pairs[FEATURES_ACCEPT].fillna(0)

pairs["eta_pred_sec"]  = eta_model.predict(X_eta)
pairs["p_accept_pred"] = accept_model.predict(X_accept).clip(0, 1)

print(f"  Mean predicted ETA    : {pairs['eta_pred_sec'].mean():.0f}s "
      f"({pairs['eta_pred_sec'].mean()/60:.1f} min)")
print(f"  Mean P_accept         : {pairs['p_accept_pred'].mean():.3f}")


# =============================================================================
# 9. COMPUTE SCORE(i, j)
# Score(i,j) = -alpha * ETA  + beta * P_accept  - gamma * d_pickup
#              - delta * courier_load  + theta * queue_urgency
# The -delta term correctly penalises assigning more orders to busy couriers.
# =============================================================================
print("\nComputing Score(i, j)...")

pairs["score"] = (
    - weights["alpha"] * pairs["eta_pred_sec"]
    + weights["beta"]  * pairs["p_accept_pred"]
    - weights["gamma"] * pairs["d_pickup_km"]
    - weights["delta"] * pairs["courier_load"]
    + weights["theta"] * pairs["queue_urgency_sec"]
)

print(f"  Score range : [{pairs['score'].min():.4f}, {pairs['score'].max():.4f}]")
print(f"  Score mean  : {pairs['score'].mean():.4f}")


# =============================================================================
# 10. ILP ASSIGNMENT (Hungarian + capacity-slot expansion)
# Each courier is replicated capacity_j times so the one-to-one Hungarian
# algorithm can assign up to capacity_j orders per courier.
# =============================================================================
print("\nSolving assignment (Hungarian + capacity expansion)...")

score_cols, slot_map = [], []
for _, cour in couriers.iterrows():
    cap       = max(1, int(cour["capacity"]))
    col_scores = pairs[pairs["courier_id"] == cour["courier_id"]]["score"].values
    if len(col_scores) != n_orders:
        col_scores = np.full(n_orders, -1e9)
    for slot in range(cap):
        score_cols.append(col_scores)
        slot_map.append({"courier_id": cour["courier_id"], "slot": slot, "capacity": cap})

score_matrix = np.column_stack(score_cols)
print(f"  Score matrix : {score_matrix.shape}")

order_idx, slot_idx = linear_sum_assignment(-score_matrix)

assignments = []
for o_idx, s_idx in zip(order_idx, slot_idx):
    if score_matrix[o_idx, s_idx] < -1e8:
        continue
    order    = orders.iloc[o_idx]
    slot     = slot_map[s_idx]
    pair_row = pairs[
        (pairs["order_id"]   == order["order_id"]) &
        (pairs["courier_id"] == slot["courier_id"])
    ].iloc[0]
    assignments.append({
        "order_id":          order["order_id"],
        "courier_id":        slot["courier_id"],
        "score":             pair_row["score"],
        "eta_pred_sec":      pair_row["eta_pred_sec"],
        "eta_pred_min":      pair_row["eta_pred_sec"] / 60,
        "p_accept":          pair_row["p_accept_pred"],
        "d_pickup_km":       pair_row["d_pickup_km"],
        "courier_load":      pair_row["courier_load"],
        "queue_urgency_sec": pair_row["queue_urgency_sec"],
        "capacity_j":        slot["capacity"],
    })

assignments_df = pd.DataFrame(assignments)
print(f"\n  Orders assigned : {len(assignments_df)} / {n_orders}")
print(f"  Total score     : {assignments_df['score'].sum():.4f}")
print(f"  Mean ETA        : {assignments_df['eta_pred_min'].mean():.1f} min")
print(f"  Mean P_accept   : {assignments_df['p_accept'].mean():.3f}")
print(f"  Mean d_pickup   : {assignments_df['d_pickup_km'].mean():.3f} km")


# =============================================================================
# 11. VERIFY CONSTRAINTS
# =============================================================================
print("\nVerifying constraints...")

order_counts = assignments_df["order_id"].value_counts()
print(f"  Order uniqueness violations : {(order_counts > 1).sum()}  (target: 0)")

load_check = (
    assignments_df.groupby("courier_id")
    .agg(assigned=("order_id", "count"), capacity_j=("capacity_j", "first"))
    .reset_index()
)
load_check["violated"] = load_check["assigned"] > load_check["capacity_j"]
print(f"  Capacity violations         : {load_check['violated'].sum()}  (target: 0)")
print(load_check.to_string(index=False))


# =============================================================================
# 12. SAVE
# =============================================================================
assignments_df.to_csv(QUEUE_DIR / "assignments.csv", index=False)
print(f"\nSaved: assignments.csv")

print("\n" + "=" * 55)
print("FINAL ASSIGNMENT SAMPLE")
print("=" * 55)
print(assignments_df[[
    "order_id", "courier_id", "score", "eta_pred_min",
    "p_accept", "d_pickup_km", "queue_urgency_sec",
]].head(15).round(3).to_string(index=False))
print("\nDone.")
