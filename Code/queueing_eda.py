"""
Food Delivery Dispatching -- Queueing EDA
==========================================
Run AFTER queueing_theory.py (needs its 3 output CSVs).
Generates 5 plots saved to an eda_plots/ subfolder.

Note: all dispatch checkpoints fall within 03:25-03:30 each day, so there
is no intra-day hour variation. Plot 1 therefore shows lambda trend by date
instead of by hour.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATA_DIR = Path(__file__).parent          # CSVs written here by queueing_theory.py
OUT_DIR  = DATA_DIR / "eda_plots"
OUT_DIR.mkdir(exist_ok=True)

C_BLUE   = "#1a73e8"
C_ORANGE = "#e8711a"
C_RED    = "#d93025"
C_GREEN  = "#1e8e3e"
C_GRAY   = "#5f6368"


def zone_to_coords(zone_id_series, grid_size=0.01):
    """Reconstruct zone centroid from zone_id string 'lat_bin_lng_bin'."""
    split   = zone_id_series.str.split("_", expand=True).astype(float)
    lat_c   = (split[0] + 0.5) * grid_size
    lng_c   = (split[1] + 0.5) * grid_size
    return lat_c, lng_c


# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
ggc     = pd.read_csv(DATA_DIR / "zone_date_queueing.csv", parse_dates=["date"])
urgency = pd.read_csv(DATA_DIR / "orders_with_urgency.csv", parse_dates=["dispatch_time"])
print("  Loaded. Generating plots...")


# =============================================================================
# PLOT 1: Lambda trend by date
# (Hour-of-day plots are uninformative since all data is at 03:25-03:30 AM.)
# =============================================================================
daily_lambda = (
    ggc.groupby("date")["lambda_per_min"]
    .agg(mean="mean", median="median",
         q25=lambda x: x.quantile(0.25),
         q75=lambda x: x.quantile(0.75))
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 5))
dates = daily_lambda["date"]
ax.fill_between(dates, daily_lambda["q25"], daily_lambda["q75"],
                alpha=0.2, color=C_BLUE, label="IQR (25-75th pct)")
ax.plot(dates, daily_lambda["median"], color=C_BLUE, lw=2, label="Median lambda")
ax.plot(dates, daily_lambda["mean"],   color=C_ORANGE, lw=1.5, ls="--", label="Mean lambda")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("lambda (orders / min per zone)", fontsize=12)
ax.set_title("Order Arrival Rate (lambda) by Date", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_lambda_by_date.png", dpi=150)
plt.close()
print("  Saved: 01_lambda_by_date.png")


# =============================================================================
# PLOT 2: Spatial heatmap of mean lambda
# =============================================================================
ggc_agg = ggc.groupby("zone_id")["lambda_per_min"].mean().reset_index()
lat_c, lng_c = zone_to_coords(ggc_agg["zone_id"])
ggc_agg["lat_c"] = lat_c
ggc_agg["lng_c"] = lng_c

fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(
    ggc_agg["lng_c"], ggc_agg["lat_c"],
    c=ggc_agg["lambda_per_min"], s=40, alpha=0.7, cmap="YlOrRd",
    vmin=0, vmax=ggc_agg["lambda_per_min"].quantile(0.95),
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Mean lambda (orders/min)", fontsize=11)
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.set_title("Spatial Distribution of Order Arrival Rate\n(grid cell average, 0.01° grid)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_lambda_spatial_heatmap.png", dpi=150)
plt.close()
print("  Saved: 02_lambda_spatial_heatmap.png")


# =============================================================================
# PLOT 3: Queue wait time distribution (clipped at 30 min for display)
# =============================================================================
wait_min       = urgency["queue_urgency_min"]
wait_min_clip  = wait_min.clip(upper=30)
clip_pct       = (wait_min > 30).mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(wait_min_clip, bins=60, color=C_BLUE, edgecolor="white", linewidth=0.3)
axes[0].axvline(wait_min_clip.median(), color=C_RED, lw=1.5, ls="--",
                label=f"Median: {wait_min_clip.median():.1f} min")
axes[0].axvline(wait_min_clip.quantile(0.9), color=C_ORANGE, lw=1.5, ls="--",
                label=f"90th pct: {wait_min_clip.quantile(0.9):.1f} min")
axes[0].set_xlabel("Queue wait time, min (clipped at 30 min)", fontsize=11)
axes[0].set_ylabel("Count", fontsize=11)
axes[0].set_title(f"Distribution of Queue Wait Times\n({clip_pct:.1%} of orders clipped at 30 min)",
                  fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

sorted_w = np.sort(wait_min_clip)
cdf = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
axes[1].plot(sorted_w, cdf, color=C_BLUE, lw=2)
for p, label in [(0.5, "50%"), (0.9, "90%"), (0.95, "95%")]:
    axes[1].axhline(p, color=C_GRAY, lw=0.8, ls=":")
    axes[1].text(30.5, p, label, va="center", fontsize=8, color=C_GRAY)
axes[1].set_xlabel("Queue wait time (min, clipped at 30 min)", fontsize=11)
axes[1].set_ylabel("Cumulative probability", fontsize=11)
axes[1].set_title("CDF of Queue Wait Times", fontsize=12, fontweight="bold")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
axes[1].grid(alpha=0.3)

plt.suptitle(
    f"Orders waiting > 5 min: {(wait_min > 5).mean():.1%}  |  "
    f"waiting > 2 min: {(wait_min > 2).mean():.1%}",
    fontsize=10, color=C_GRAY,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_queue_wait_distribution.png", dpi=150)
plt.close()
print("  Saved: 03_queue_wait_distribution.png")


# =============================================================================
# PLOT 4: Courier utilization rho
# =============================================================================
rho          = ggc["rho"].clip(upper=2)
rho_raw      = ggc["rho_raw"] if "rho_raw" in ggc.columns else ggc["rho"]
n_overloaded = (ggc.get("overloaded", ggc["rho"] >= 1)).sum()
pct_over     = n_overloaded / len(ggc)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(rho[rho < 1],  bins=40, color=C_BLUE, edgecolor="white", linewidth=0.3,
             label="Stable (rho < 1)", alpha=0.8)
axes[0].hist(rho[rho >= 1], bins=5,  color=C_RED,  edgecolor="white", linewidth=0.3,
             label=f"Overloaded (rho >= 1): {pct_over:.1%}", alpha=0.8)
axes[0].axvline(1.0, color=C_RED, lw=1.5, ls="--", label="Stability threshold")
axes[0].axvline(ggc["rho"].median(), color=C_ORANGE, lw=1.5, ls="--",
                label=f"Median rho = {ggc['rho'].median():.2f}")
axes[0].set_xlabel("System utilization rho = lambda / (c * mu)", fontsize=11)
axes[0].set_ylabel("Zone-date count", fontsize=11)
axes[0].set_title("G/G/c Utilization (rho) Distribution", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

ggc["date_str"] = pd.to_datetime(ggc["date"]).dt.strftime("%m/%d")
rho_by_date = ggc.groupby("date_str")["rho"].median().reset_index()
bar_colors = [C_RED if r >= 0.9 else C_BLUE for r in rho_by_date["rho"]]
axes[1].bar(rho_by_date["date_str"], rho_by_date["rho"], color=bar_colors, alpha=0.8)
axes[1].axhline(1.0, color=C_RED, lw=1.5, ls="--", label="Stability threshold (rho=1)")
axes[1].set_xlabel("Date", fontsize=11)
axes[1].set_ylabel("Median rho", fontsize=11)
axes[1].set_title("Median Utilization (rho) by Date", fontsize=12, fontweight="bold")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "04_utilization_rho.png", dpi=150)
plt.close()
print("  Saved: 04_utilization_rho.png")


# =============================================================================
# PLOT 5: Capacity from Little's Law
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cap_counts = ggc["capacity"].value_counts().sort_index()
axes[0].bar(cap_counts.index, cap_counts.values, color=C_BLUE, edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Capacity per courier (orders)", fontsize=11)
axes[0].set_ylabel("Zone-date count", fontsize=11)
axes[0].set_title("Dynamic Courier Capacity from Little's Law", fontsize=12, fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

axes[1].scatter(ggc["rho"].clip(upper=1.5), ggc["L_per_courier"].clip(upper=10),
                alpha=0.5, s=20, color=C_BLUE)
axes[1].axvline(1.0, color=C_RED, lw=1, ls="--", label="rho = 1 (instability)")
axes[1].set_xlabel("Utilization rho (clipped at 1.5)", fontsize=11)
axes[1].set_ylabel("L = lambda * W (avg orders per courier)", fontsize=11)
axes[1].set_title("Little's Law: Orders in System vs Utilization", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.suptitle(
    f"Mean capacity = {ggc['capacity'].mean():.1f}  |  L = (lambda/c) * W_service",
    fontsize=10, color=C_GRAY,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_capacity_littles_law.png", dpi=150)
plt.close()
print("  Saved: 05_capacity_littles_law.png")

print(f"\nAll plots saved to: {OUT_DIR}")
print("Done.")
