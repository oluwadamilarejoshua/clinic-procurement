"""
EHA Clinics - Procurement Predictive Modelling
Exploratory Data Analysis Script
Generates all figures for the EDA report.
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Project paths (resolved relative to this file, works from any cwd) ────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Brand colours ─────────────────────────────────────────────────────────────
BLUE       = "#00548E"
LIGHT_BLUE = "#E0EDF8"
GREEN      = "#009976"
GREY       = "#F5F5F5"
DARK_GREY  = "#4A4A4A"
RED        = "#C0392B"
AMBER      = "#E67E22"
PALETTE    = [BLUE, GREEN, "#E74C3C", "#F39C12", "#8E44AD",
              "#1ABC9C", "#2C3E50", "#D35400", "#7F8C8D"]

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.titlecolor":  BLUE,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
})

# ── Constants ─────────────────────────────────────────────────────────────────
INSCOPE_CATS = {
    "Prescription Medications", "Over The Counter Drugs",
    "Vaccines", "NPI Vaccines", "Injections",
    "Medical Consumables", "Laboratory Consumables",
    "Dental Consumables", "Consumables",
}

PILOT_FACILITIES = {
    "Abuja - Asba and Dantata",
    "Abuja - Lugbe",
    "Kano - Lamido Crescent",
    "Kano - Independence Road",
    "Lagos - Sangotedo Ajah",
    "REACH Abuja Hub 1 Clinic - Kuje",
}

SHORT_NAMES = {
    "Abuja - Asba and Dantata":          "Asba (Abuja)",
    "Abuja - Lugbe":                     "Lugbe (Abuja)",
    "Kano - Lamido Crescent":            "Lamido (Kano)",
    "Kano - Independence Road":          "Indep. Rd (Kano)",
    "Lagos - Sangotedo Ajah":            "Sangotedo (Lagos)",
    "REACH Abuja Hub 1 Clinic - Kuje":   "REACH Kuje",
}

SHORT_CATS = {
    "Prescription Medications": "Rx Meds",
    "Over The Counter Drugs":   "OTC Drugs",
    "Medical Consumables":      "Med. Consumables",
    "Laboratory Consumables":   "Lab Consumables",
    "Consumables":              "Consumables",
    "Injections":               "Injections",
    "Vaccines":                 "Vaccines",
    "NPI Vaccines":             "NPI Vaccines",
    "Dental Consumables":       "Dental Consumables",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df_raw = pd.read_csv(DATA_DIR / "Clinics Procurement Data.csv", low_memory=False)
df_raw["date_order"] = pd.to_datetime(df_raw["date_order"], errors="coerce")
df_raw["month"]      = pd.to_datetime(df_raw["month"], errors="coerce")
df_raw["product_qty"]      = pd.to_numeric(df_raw["product_qty"], errors="coerce").fillna(0)
df_raw["price_subtotal"]   = pd.to_numeric(df_raw["price_subtotal"], errors="coerce").fillna(0)
df_raw["year"]             = pd.to_numeric(df_raw["year"], errors="coerce")

# Cleaned working dataset: done orders only
df_done = df_raw[df_raw["state"] == "done"].copy()

# In-scope pilot dataset
df_scope = df_done[
    df_done["category_name"].isin(INSCOPE_CATS) &
    df_done["branch_name"].isin(PILOT_FACILITIES)
].copy()
df_scope["branch_short"] = df_scope["branch_name"].map(SHORT_NAMES)
df_scope["cat_short"]    = df_scope["category_name"].map(SHORT_CATS)

print(f"  Raw rows       : {len(df_raw):,}")
print(f"  Done rows      : {len(df_done):,}")
print(f"  In-scope rows  : {len(df_scope):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — DATA QUALITY OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1: Data quality...")

total = len(df_raw)
issues = {
    "Missing branch name":      (df_raw["branch_name"].isna() | (df_raw["branch_name"] == "")).sum(),
    "Zero-quantity records":    (df_raw["product_qty"] == 0).sum(),
    "Negative spend records":   (df_raw["price_subtotal"] < 0).sum(),
    "Pending orders (purchase state)": (df_raw["state"] == "purchase").sum(),
    "Missing order date":       df_raw["date_order"].isna().sum(),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: absolute counts
labels  = list(issues.keys())
counts  = list(issues.values())
colours = [AMBER if c > 1000 else GREEN for c in counts]
bars = axes[0].barh(labels, counts, color=colours, edgecolor="white", height=0.55)
for bar, cnt in zip(bars, counts):
    axes[0].text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                 f"{cnt:,}", va="center", fontsize=9, color=DARK_GREY)
axes[0].set_xlabel("Record count")
axes[0].set_title("Data Quality Issues — Absolute Count")
axes[0].invert_yaxis()

# Right: as % of total
pcts = [c / total * 100 for c in counts]
bars2 = axes[1].barh(labels, pcts, color=colours, edgecolor="white", height=0.55)
for bar, pct in zip(bars2, pcts):
    axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{pct:.1f}%", va="center", fontsize=9, color=DARK_GREY)
axes[1].set_xlabel("Percentage of total records")
axes[1].set_title("Data Quality Issues — As % of Total")
axes[1].invert_yaxis()

fig.suptitle("Dataset Quality Assessment  (N = 43,799 rows)", fontsize=13,
             fontweight="bold", color=BLUE, y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_data_quality.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — ANNUAL PROCUREMENT TREND (in-scope, pilot)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Annual trend...")

annual = (
    df_scope.groupby("year")
    .agg(total_qty=("product_qty", "sum"),
         total_spend=("price_subtotal", "sum"),
         order_count=("order_id", "nunique"))
    .reset_index()
)
annual = annual[annual["year"].between(2020, 2024)]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)

bars = ax1.bar(annual["year"].astype(int), annual["total_qty"],
               color=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.2, label="Total Quantity")
line = ax2.plot(annual["year"].astype(int), annual["total_spend"] / 1e6,
                color=GREEN, marker="o", linewidth=2.2, label="Total Spend (M NGN)")

ax1.set_xlabel("Year")
ax1.set_ylabel("Total Units Procured", color=BLUE)
ax2.set_ylabel("Total Spend (Million NGN)", color=GREEN)
ax1.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=GREEN)

# Annotate 2021 spike
spike_idx = annual[annual["year"] == 2021]
if not spike_idx.empty:
    y_val = spike_idx["total_qty"].values[0]
    ax1.annotate("Bulk stock-load\nevent (Sep 2021)",
                 xy=(2021, y_val), xytext=(2021.3, y_val * 0.85),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

handles = [mpatches.Patch(color=LIGHT_BLUE, label="Total Quantity"),
           plt.Line2D([0],[0], color=GREEN, marker="o", label="Total Spend (M NGN)")]
ax1.legend(handles=handles, loc="upper right", fontsize=9)
ax1.set_title("Annual In-Scope Procurement — Pilot Facilities (2020–2024)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_annual_trend.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — FACILITY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Facility comparison...")

# Exclude the 2021 Kano Independence Road bulk event for spend comparison
df_scope_clean = df_scope[~(
    (df_scope["branch_name"] == "Kano - Independence Road") &
    (df_scope["year"] == 2021)
)]

fac_stats = (
    df_scope_clean.groupby("branch_short")
    .agg(qty=("product_qty","sum"), spend=("price_subtotal","sum"),
         orders=("order_id","nunique"), months=("month","nunique"))
    .reset_index()
    .sort_values("spend", ascending=True)
)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Spend
axes[0].barh(fac_stats["branch_short"], fac_stats["spend"]/1e6,
             color=BLUE, edgecolor="white")
axes[0].set_xlabel("M NGN")
axes[0].set_title("Total Spend\n(excl. 2021 bulk event)")
for i, v in enumerate(fac_stats["spend"]/1e6):
    axes[0].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

# Quantity
axes[1].barh(fac_stats["branch_short"], fac_stats["qty"],
             color=GREEN, edgecolor="white")
axes[1].set_xlabel("Units")
axes[1].set_title("Total Quantity\n(excl. 2021 bulk event)")
for i, v in enumerate(fac_stats["qty"]):
    axes[1].text(v + 50, i, f"{v:,.0f}", va="center", fontsize=8)

# Active months
axes[2].barh(fac_stats["branch_short"], fac_stats["months"],
             color=AMBER, edgecolor="white")
axes[2].set_xlabel("Months")
axes[2].set_title("Active Procurement\nMonths")
for i, v in enumerate(fac_stats["months"]):
    axes[2].text(v + 0.3, i, str(int(v)), va="center", fontsize=8)

fig.suptitle("Pilot Facility Profile — In-Scope Health Commodities",
             fontsize=13, fontweight="bold", color=BLUE)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig3_facility_profile.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — CATEGORY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4: Category breakdown...")

df_scope_clean2 = df_scope[~(
    (df_scope["branch_name"] == "Kano - Independence Road") &
    (df_scope["year"] == 2021)
)]
cat_stats = (
    df_scope_clean2.groupby("cat_short")
    .agg(qty=("product_qty","sum"), spend=("price_subtotal","sum"))
    .reset_index()
    .sort_values("spend", ascending=True)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Spend
axes[0].barh(cat_stats["cat_short"], cat_stats["spend"]/1e6,
             color=[BLUE if i % 2 == 0 else GREEN for i in range(len(cat_stats))],
             edgecolor="white")
axes[0].set_xlabel("M NGN")
axes[0].set_title("Spend by Category (excl. 2021 bulk event)")
for i, v in enumerate(cat_stats["spend"]/1e6):
    axes[0].text(v + 0.3, i, f"{v:.1f}M", va="center", fontsize=8)

# Quantity
cat_qty_sorted = cat_stats.sort_values("qty", ascending=True)
axes[1].barh(cat_qty_sorted["cat_short"], cat_qty_sorted["qty"],
             color=[BLUE if i % 2 == 0 else GREEN for i in range(len(cat_qty_sorted))],
             edgecolor="white")
axes[1].set_xlabel("Units")
axes[1].set_title("Volume by Category (excl. 2021 bulk event)")
for i, v in enumerate(cat_qty_sorted["qty"]):
    axes[1].text(v + 100, i, f"{v:,.0f}", va="center", fontsize=8)

fig.suptitle("In-Scope Category Profile — Spend and Volume",
             fontsize=13, fontweight="bold", color=BLUE)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_category_profile.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — MONTHLY DEMAND TRENDS (top 4 categories, 2021–2024)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 5: Monthly demand trends...")

TOP_CATS = ["Prescription Medications", "Over The Counter Drugs",
            "Medical Consumables", "Vaccines"]

df_trend = df_scope[
    df_scope["category_name"].isin(TOP_CATS) &
    df_scope["year"].between(2021, 2024) &
    ~((df_scope["branch_name"] == "Kano - Independence Road") & (df_scope["year"] == 2021))
].copy()

monthly_cat = (
    df_trend.groupby(["month", "category_name"])["product_qty"]
    .sum().reset_index()
)
monthly_cat["month"] = pd.to_datetime(monthly_cat["month"])

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
axes = axes.flatten()

for i, cat in enumerate(TOP_CATS):
    sub = monthly_cat[monthly_cat["category_name"] == cat].sort_values("month")
    axes[i].fill_between(sub["month"], sub["product_qty"],
                         alpha=0.18, color=PALETTE[i])
    axes[i].plot(sub["month"], sub["product_qty"],
                 color=PALETTE[i], linewidth=1.8, marker="o", markersize=3)
    axes[i].set_title(cat)
    axes[i].set_ylabel("Units Procured")
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"))

    # Shade COVID-adjacent months
    axes[i].axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-06-01"),
                    alpha=0.07, color=AMBER, label="Post-COVID ramp" if i == 0 else "")

fig.suptitle("Monthly Procurement Volume — Top 4 Health Commodity Categories (2021–2024)",
             fontsize=12, fontweight="bold", color=BLUE)
fig.autofmt_xdate(rotation=30)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig5_monthly_trends.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — FORECAST READINESS HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 6: Readiness heatmap...")

# Count non-zero months per pair using 2021-2023 training window
df_train = df_scope[
    df_scope["year"].between(2021, 2023) &
    (df_scope["product_qty"] > 0) &
    ~((df_scope["branch_name"] == "Kano - Independence Road") & (df_scope["year"] == 2021))
].copy()

readiness = (
    df_train.groupby(["cat_short", "branch_short"])["month"]
    .nunique().reset_index()
    .rename(columns={"month": "non_zero_months"})
)
pivot = readiness.pivot(index="cat_short", columns="branch_short",
                        values="non_zero_months").fillna(0)

# Reorder columns
col_order = [SHORT_NAMES[f] for f in [
    "Abuja - Asba and Dantata", "Kano - Lamido Crescent",
    "Abuja - Lugbe", "Lagos - Sangotedo Ajah",
    "Kano - Independence Road", "REACH Abuja Hub 1 Clinic - Kuje"
] if SHORT_NAMES[f] in pivot.columns]
pivot = pivot[[c for c in col_order if c in pivot.columns]]

fig, ax = plt.subplots(figsize=(11, 5))
cmap = sns.color_palette("YlOrRd", as_cmap=True)
sns.heatmap(pivot.astype(float), ax=ax, annot=True, fmt=".0f",
            cmap=cmap, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Non-zero months (max 36)"},
            vmin=0, vmax=36)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Forecast Readiness — Non-Zero Procurement Months per Category × Facility\n"
             "(Training window: Jan 2021 – Dec 2023,  excl. Kano Indep. Rd bulk event)",
             fontsize=11, fontweight="bold", color=BLUE)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig6_readiness_heatmap.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — SEASONALITY PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 7: Seasonality...")

df_seas = df_scope[
    df_scope["year"].between(2021, 2023) &
    (df_scope["product_qty"] > 0) &
    df_scope["category_name"].isin(["Prescription Medications",
                                    "Over The Counter Drugs",
                                    "Medical Consumables", "Vaccines"]) &
    ~((df_scope["branch_name"] == "Kano - Independence Road") & (df_scope["year"] == 2021))
].copy()
df_seas["month_num"] = df_seas["date_order"].dt.month
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

monthly_agg = (
    df_seas.groupby(["category_name","month_num"])["product_qty"]
    .mean().reset_index()
)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
axes = axes.flatten()
for i, cat in enumerate(["Prescription Medications","Over The Counter Drugs",
                          "Medical Consumables","Vaccines"]):
    sub = monthly_agg[monthly_agg["category_name"] == cat].sort_values("month_num")
    ax = axes[i]
    ax.bar(sub["month_num"], sub["product_qty"],
           color=PALETTE[i], edgecolor="white", alpha=0.85)
    ax.plot(sub["month_num"], sub["product_qty"],
            color=DARK_GREY, linewidth=1.2, linestyle="--", marker=".")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_title(cat)
    ax.set_ylabel("Avg. Monthly Units")
    overall_mean = sub["product_qty"].mean()
    ax.axhline(overall_mean, color=RED, linestyle=":", linewidth=1.2,
               label=f"Annual mean: {overall_mean:,.0f}")
    ax.legend(fontsize=8)

fig.suptitle("Seasonal Demand Patterns — Average Monthly Volume by Category (2021–2023)",
             fontsize=12, fontweight="bold", color=BLUE)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig7_seasonality.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — DEMAND VOLATILITY (CV heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 8: Volatility heatmap...")

df_cv = df_scope[
    df_scope["year"].between(2021, 2023) &
    (df_scope["product_qty"] > 0) &
    ~((df_scope["branch_name"] == "Kano - Independence Road") & (df_scope["year"] == 2021))
].copy()

monthly_pair = (
    df_cv.groupby(["cat_short","branch_short","month"])["product_qty"]
    .sum().reset_index()
)
cv_df = (
    monthly_pair.groupby(["cat_short","branch_short"])["product_qty"]
    .agg(["mean","std"]).reset_index()
)
cv_df["cv"] = (cv_df["std"] / cv_df["mean"] * 100).fillna(0)

pivot_cv = cv_df.pivot(index="cat_short", columns="branch_short", values="cv").fillna(np.nan)
pivot_cv = pivot_cv[[c for c in col_order if c in pivot_cv.columns]]

fig, ax = plt.subplots(figsize=(11, 5))
mask = pivot_cv.isna()
cmap_cv = sns.diverging_palette(130, 15, as_cmap=True)
sns.heatmap(pivot_cv, ax=ax, annot=True, fmt=".0f",
            cmap="RdYlGn_r", linewidths=0.5, linecolor="white",
            mask=mask, cbar_kws={"label": "CV (%) — lower is more stable"},
            vmin=0, vmax=150)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Demand Volatility — Coefficient of Variation (%) per Category × Facility\n"
             "(Training window: Jan 2021 – Dec 2023,  lower = more forecastable)",
             fontsize=11, fontweight="bold", color=BLUE)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig8_volatility_heatmap.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — TOP 20 PRODUCTS BY QUANTITY
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 9: Top products...")

df_scope_clean3 = df_scope[~(
    (df_scope["branch_name"] == "Kano - Independence Road") &
    (df_scope["year"] == 2021)
)]
top_prods = (
    df_scope_clean3.groupby(["product_name","category_name"])["product_qty"]
    .sum().reset_index()
    .sort_values("product_qty", ascending=False)
    .head(20)
)

# Truncate long names
top_prods["prod_short"] = top_prods["product_name"].str[:52]

cat_colour_map = {c: PALETTE[i % len(PALETTE)] for i, c in
                  enumerate(top_prods["category_name"].unique())}
colours = top_prods["category_name"].map(cat_colour_map)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top_prods["prod_short"][::-1].values,
               top_prods["product_qty"][::-1].values,
               color=list(reversed(list(colours))), edgecolor="white")
for bar, v in zip(bars, top_prods["product_qty"][::-1].values):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f"{v:,.0f}", va="center", fontsize=8)

legend_patches = [mpatches.Patch(color=cat_colour_map[c], label=c)
                  for c in cat_colour_map]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
ax.set_xlabel("Total Units Procured")
ax.set_title("Top 20 Products by Volume — In-Scope Categories, Pilot Facilities\n"
             "(excl. Kano Independence Rd 2021 bulk event)",
             fontsize=11, fontweight="bold", color=BLUE)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig9_top_products.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — KANO INDEPENDENCE ROAD ANOMALY DRILL-DOWN
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 10: Kano anomaly drill-down...")

df_kano = df_scope[df_scope["branch_name"] == "Kano - Independence Road"].copy()
kano_monthly = (
    df_kano.groupby("month")["product_qty"].sum().reset_index()
)
kano_monthly["month"] = pd.to_datetime(kano_monthly["month"])
kano_monthly = kano_monthly.sort_values("month")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Full monthly series
axes[0].fill_between(kano_monthly["month"], kano_monthly["product_qty"],
                     alpha=0.2, color=RED)
axes[0].plot(kano_monthly["month"], kano_monthly["product_qty"],
             color=RED, linewidth=1.6)
axes[0].axvspan(pd.Timestamp("2021-08-01"), pd.Timestamp("2021-10-01"),
                alpha=0.2, color=AMBER, label="Bulk event window")
axes[0].legend(fontsize=9)
axes[0].set_title("Kano – Independence Rd: Full Monthly Series")
axes[0].set_ylabel("Units Procured")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].tick_params(axis="x", rotation=30)

# Sep 2021 product breakdown
df_sep21 = df_scope[
    (df_scope["branch_name"] == "Kano - Independence Road") &
    (df_scope["month"].dt.to_period("M") == "2021-09")
]
sep_prods = (
    df_sep21.groupby("product_name")["product_qty"]
    .sum().sort_values(ascending=True).tail(10)
)
sep_short = sep_prods.copy()
sep_short.index = [n[:42] for n in sep_short.index]

axes[1].barh(sep_short.index, sep_short.values,
             color=RED, alpha=0.75, edgecolor="white")
for i, v in enumerate(sep_short.values):
    axes[1].text(v + 200, i, f"{v:,.0f}", va="center", fontsize=8)
axes[1].set_xlabel("Units")
axes[1].set_title("September 2021 — Top 10 Products Driving the Spike")

fig.suptitle("Kano Independence Rd: Bulk Stock-Load Event Analysis (September 2021)",
             fontsize=12, fontweight="bold", color=RED)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig10_kano_anomaly.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS (saved for reference)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nAll figures saved to", FIGURES_DIR)
print("\nKey summary statistics:")
print(f"  Total raw records        : {len(df_raw):,}")
print(f"  Confirmed done orders    : {len(df_done):,}")
print(f"  In-scope pilot records   : {len(df_scope):,}")
print(f"  Unique in-scope products : {df_scope['product_name'].nunique():,}")
date_min = df_raw["date_order"].min().date()
date_max = df_raw["date_order"].max().date()
print(f"  Date range (all data)    : {date_min} to {date_max}")
print(f"  Total in-scope spend     : NGN {df_scope['price_subtotal'].sum()/1e6:.1f}M")
