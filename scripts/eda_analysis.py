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
    "Asba & Dantata, Abuja",
    "Abuja - Lugbe",
    "Kano - Lamido Crescent",
    "Kano - Independence Road",
    "Lagos- Sangotedo Ajah",
    "REACH Abuja Hub 1 Clinic - Kuje",
}

SHORT_NAMES = {
    "Asba & Dantata, Abuja":           "Asba (Abuja)",
    "Abuja - Lugbe":                   "Lugbe (Abuja)",
    "Kano - Lamido Crescent":          "Lamido (Kano)",
    "Kano - Independence Road":        "Indep. Rd (Kano)",
    "Lagos- Sangotedo Ajah":           "Sangotedo (Lagos)",
    "REACH Abuja Hub 1 Clinic - Kuje": "REACH Kuje",
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
df_raw = pd.read_csv(DATA_DIR / "Clinics Procurement Data-1778233381741.csv", low_memory=False)
df_raw = df_raw.rename(columns={
    "product_uom_qty":   "product_qty",
    "product_category":  "category_name",
    "requesting_branch": "branch_name",
    "product":           "product_name",
    "purchase_order":    "order_id",
})
df_raw["date_order"]     = pd.to_datetime(df_raw["date_order"], errors="coerce")
df_raw["month"]          = df_raw["date_order"].dt.to_period("M").dt.to_timestamp()
df_raw["year"]           = df_raw["date_order"].dt.year
df_raw["product_qty"]    = pd.to_numeric(df_raw["product_qty"], errors="coerce").fillna(0)
df_raw["price_subtotal"] = pd.to_numeric(df_raw["price_subtotal"], errors="coerce").fillna(0)

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

fig.suptitle("Dataset Quality Assessment  (N = 34,300 rows)", fontsize=13,
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
annual = annual[annual["year"].between(2020, 2025)]

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
ax1.set_title("Annual In-Scope Procurement — Pilot Facilities (2020–2025)")
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
    "Asba & Dantata, Abuja", "Kano - Lamido Crescent",
    "Abuja - Lugbe", "Lagos- Sangotedo Ajah",
    "Kano - Independence Road", "REACH Abuja Hub 1 Clinic - Kuje"
] if f in SHORT_NAMES and SHORT_NAMES[f] in pivot.columns]
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
# ─── MODEL PERFORMANCE DIAGNOSTIC CHECKS  (Figs 11 – 15) ─────────────────────
#
# These five figures diagnose the structural data-quality reasons behind
# forecasting model underperformance, covering:
#   • demand type / intermittency (Syntetos-Boylan framework)
#   • zero-inflation rates that break standard time-series models
#   • distribution shift between the 2021-23 training window and 2024 H1
#     validation period (relevant to all four modelling tiers)
#   • outlier contamination that biases recursive ML lag-feature models
#   • training-data sufficiency relative to model minimum-sample thresholds
# ═══════════════════════════════════════════════════════════════════════════════

# ── Shared pre-computation: build complete 36-month monthly demand grid ────────
# Training window: Jan 2021 – Dec 2023, Kano bulk event excluded throughout
_df_diag = df_scope[
    df_scope["year"].between(2021, 2023) &
    ~((df_scope["branch_name"] == "Kano - Independence Road") &
      (df_scope["year"] == 2021))
].copy()

_all_months = pd.date_range("2021-01-01", "2023-12-01", freq="MS")
_pairs = (
    _df_diag.groupby(["cat_short", "branch_short"])
    .size().reset_index()[["cat_short", "branch_short"]]
)

_intermittency_records = []
for _, row in _pairs.iterrows():
    sub = _df_diag[
        (_df_diag["cat_short"]    == row["cat_short"]) &
        (_df_diag["branch_short"] == row["branch_short"])
    ]
    # Aggregate to monthly totals and fill the full 36-month calendar
    monthly = (
        sub.groupby(pd.Grouper(key="month", freq="MS"))["product_qty"]
        .sum()
        .reindex(_all_months, fill_value=0)
    )
    n_total   = len(monthly)         # always 36
    n_nonzero = int((monthly > 0).sum())
    if n_nonzero == 0:
        continue
    adi = n_total / n_nonzero        # Average Demand Interval
    nz  = monthly[monthly > 0]
    cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else 0.0
    _intermittency_records.append({
        "cat_short":    row["cat_short"],
        "branch_short": row["branch_short"],
        "adi":          adi,
        "cv2":          cv2,
        "n_months":     n_total,
        "n_nonzero":    n_nonzero,
        "zero_rate":    (n_total - n_nonzero) / n_total,
    })

intermittency_df = pd.DataFrame(_intermittency_records)

# Syntetos-Boylan (2005) classification thresholds
ADI_THRESH = 1.32
CV2_THRESH = 0.49

def _classify_demand(adi, cv2):
    if   adi < ADI_THRESH and cv2 < CV2_THRESH: return "Smooth"
    elif adi < ADI_THRESH:                       return "Erratic"
    elif                    cv2 < CV2_THRESH:    return "Intermittent"
    else:                                        return "Lumpy"

intermittency_df["demand_class"] = intermittency_df.apply(
    lambda r: _classify_demand(r["adi"], r["cv2"]), axis=1
)

CLASS_COLOURS = {
    "Smooth":       GREEN,
    "Erratic":      AMBER,
    "Intermittent": BLUE,
    "Lumpy":        RED,
}
CLASS_ORDER = ["Smooth", "Erratic", "Intermittent", "Lumpy"]


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — DEMAND INTERMITTENCY CLASSIFICATION (Syntetos-Boylan)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 11: Demand intermittency classification...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left — ADI vs CV² scatter, colour-coded by demand class
for cls, grp in intermittency_df.groupby("demand_class"):
    axes[0].scatter(
        grp["adi"], grp["cv2"],
        c=CLASS_COLOURS[cls], label=cls,
        s=85, alpha=0.85, edgecolors="white", linewidths=0.5,
    )

axes[0].axvline(ADI_THRESH, color=DARK_GREY, linestyle="--", linewidth=1, alpha=0.5)
axes[0].axhline(CV2_THRESH, color=DARK_GREY, linestyle="--", linewidth=1, alpha=0.5)

# Quadrant labels anchored to threshold lines
axes[0].text(ADI_THRESH * 0.97, CV2_THRESH * 0.12,
             "Smooth\n(best for ETS/SARIMA)", fontsize=7.5,
             color=GREEN, ha="right", va="bottom")
axes[0].text(ADI_THRESH * 0.97, CV2_THRESH * 1.7,
             "Erratic\n(high variance)", fontsize=7.5,
             color=AMBER, ha="right", va="bottom")
axes[0].text(ADI_THRESH * 1.03, CV2_THRESH * 0.12,
             "Intermittent\n(sparse demand)", fontsize=7.5,
             color=BLUE, ha="left", va="bottom")
axes[0].text(ADI_THRESH * 1.03, CV2_THRESH * 1.7,
             "Lumpy\n(sparse + volatile)", fontsize=7.5,
             color=RED, ha="left", va="bottom")

axes[0].set_xlabel("Average Demand Interval (ADI)  [higher → more intermittent]")
axes[0].set_ylabel("Squared Coefficient of Variation (CV²)  [higher → more variable]")
axes[0].set_title(
    "Demand Type — Syntetos-Boylan Classification\n"
    f"(ADI threshold = {ADI_THRESH},  CV² threshold = {CV2_THRESH})"
)
axes[0].legend(title="Demand Class", fontsize=9, title_fontsize=9)

# Right — class distribution bar chart
class_counts = (
    intermittency_df["demand_class"]
    .value_counts()
    .reindex(CLASS_ORDER, fill_value=0)
)
_total_pairs = len(intermittency_df)
bars = axes[1].bar(
    class_counts.index, class_counts.values,
    color=[CLASS_COLOURS[c] for c in class_counts.index],
    edgecolor="white", width=0.55,
)
for bar, cnt in zip(bars, class_counts.values):
    pct = cnt / _total_pairs * 100
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3, str(int(cnt)),
        ha="center", fontsize=10, fontweight="bold", color=DARK_GREY,
    )
    if cnt > 0:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:.0f}%", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold",
        )
axes[1].set_ylim(0, class_counts.max() * 1.28)
axes[1].set_ylabel("Number of Category × Facility Pairs")
axes[1].set_title(
    "Proportion of Pairs by Demand Class\n"
    "(training window: Jan 2021 – Dec 2023)"
)

fig.suptitle(
    "Demand Intermittency Analysis — Primary Root Cause of Forecast Model Underperformance",
    fontsize=12, fontweight="bold", color=BLUE,
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig11_intermittency_classification.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12 — ZERO-INFLATION RATE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 12: Zero-inflation heatmap...")

zero_pivot = (
    intermittency_df
    .pivot(index="cat_short", columns="branch_short", values="zero_rate")
    .mul(100)
)
zero_pivot = zero_pivot[[c for c in col_order if c in zero_pivot.columns]]

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(
    zero_pivot, ax=ax, annot=True, fmt=".0f",
    cmap="YlOrRd", linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Zero-demand months (%)"},
    vmin=0, vmax=100,
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title(
    "Zero-Inflation Rate — % of Training Months with Zero Procurement per Category × Facility\n"
    "(Training window: Jan 2021 – Dec 2023  |  >50% → standard time-series models unreliable)",
    fontsize=11, fontweight="bold", color=BLUE,
)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig12_zero_inflation.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 13 — TRAIN vs VALIDATION DISTRIBUTION SHIFT
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 13: Train–validation distribution shift...")

# Validation window used in all four modelling tiers: Jan–Jun 2024
_df_val = df_scope[
    (df_scope["month"] >= pd.Timestamp("2024-01-01")) &
    (df_scope["month"] <= pd.Timestamp("2024-06-01"))
].copy()

_train_monthly = (
    _df_diag
    .groupby(["category_name", pd.Grouper(key="month", freq="MS")])["product_qty"]
    .sum().reset_index()
)
_val_monthly = (
    _df_val
    .groupby(["category_name", pd.Grouper(key="month", freq="MS")])["product_qty"]
    .sum().reset_index()
)

_TOP4 = [
    "Prescription Medications", "Over The Counter Drugs",
    "Medical Consumables", "Vaccines",
]

shift_records = []
for cat in _TOP4:
    tr = _train_monthly[_train_monthly["category_name"] == cat]["product_qty"]
    te = _val_monthly[  _val_monthly["category_name"]   == cat]["product_qty"]
    if len(te) == 0 or tr.mean() == 0:
        continue
    shift_records.append({
        "category":       SHORT_CATS.get(cat, cat),
        "train_mean":     tr.mean(),
        "val_mean":       te.mean(),
        "train_std":      tr.std(),
        "val_std":        te.std(),
        "mean_shift_pct": (te.mean() - tr.mean()) / tr.mean() * 100,
    })
shift_df = pd.DataFrame(shift_records)

if len(shift_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(shift_df))
    w = 0.35

    # Panel 1 — Mean monthly demand
    axes[0].bar(x - w/2, shift_df["train_mean"], w,
                label="Train 2021–23", color=BLUE,  alpha=0.85, edgecolor="white")
    axes[0].bar(x + w/2, shift_df["val_mean"],   w,
                label="Val 2024 H1",   color=AMBER, alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(shift_df["category"], rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean Monthly Units")
    axes[0].set_title("Mean Demand\nTrain vs Validation")
    axes[0].legend(fontsize=8)

    # Panel 2 — Standard deviation
    axes[1].bar(x - w/2, shift_df["train_std"], w,
                label="Train 2021–23", color=BLUE,  alpha=0.85, edgecolor="white")
    axes[1].bar(x + w/2, shift_df["val_std"],   w,
                label="Val 2024 H1",   color=AMBER, alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(shift_df["category"], rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("Std Dev of Monthly Units")
    axes[1].set_title("Demand Variability\nTrain vs Validation")
    axes[1].legend(fontsize=8)

    # Panel 3 — % mean shift
    shift_cols = [RED if abs(v) > 20 else GREEN for v in shift_df["mean_shift_pct"]]
    axes[2].bar(x, shift_df["mean_shift_pct"], color=shift_cols,
                edgecolor="white", width=0.55)
    axes[2].axhline(0,   color=DARK_GREY, linewidth=1)
    axes[2].axhline( 20, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.65)
    axes[2].axhline(-20, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.65,
                    label="±20% alert threshold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(shift_df["category"], rotation=15, ha="right", fontsize=8)
    axes[2].set_ylabel("Mean Shift (%)")
    axes[2].set_title("Distribution Shift\n(Validation Mean vs Training Mean)")
    for xi, v in zip(x, shift_df["mean_shift_pct"]):
        axes[2].text(xi, v + (2 if v >= 0 else -4),
                     f"{v:+.0f}%", ha="center", fontsize=9, color=DARK_GREY)
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "Train–Validation Distribution Shift — Demand Level Change from 2021-23 Training to 2024 H1 Validation",
        fontsize=12, fontweight="bold", color=BLUE,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig13_distribution_shift.pdf", bbox_inches="tight")
    plt.close()
else:
    print("  Skipping Fig 13 — no 2024 validation data found in the dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 14 — OUTLIER CONTAMINATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 14: Outlier contamination...")

_monthly_outlier = (
    _df_diag
    .groupby(["cat_short", "branch_short", pd.Grouper(key="month", freq="MS")])
    ["product_qty"].sum().reset_index()
)

outlier_records = []
for (cat, branch), grp in _monthly_outlier.groupby(["cat_short", "branch_short"]):
    vals = grp["product_qty"]
    if len(vals) < 4:
        continue
    Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
    IQR    = Q3 - Q1
    upper  = Q3 + 3.0 * IQR
    n_out  = int((vals > upper).sum())
    mean_all = vals.mean()
    mask_in  = vals <= upper
    mean_in  = vals[mask_in].mean() if mask_in.any() else mean_all
    pct_bias = abs(mean_all - mean_in) / mean_all * 100 if mean_all > 0 else 0.0
    outlier_records.append({
        "cat_short":    cat,
        "branch_short": branch,
        "n_months":     len(vals),
        "n_outliers":   n_out,
        "outlier_pct":  n_out / len(vals) * 100,
        "mean_bias_pct": pct_bias,
    })

outlier_df = pd.DataFrame(outlier_records)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left — outlier rate heatmap
out_pivot = (
    outlier_df
    .pivot(index="cat_short", columns="branch_short", values="outlier_pct")
    .fillna(0)
)
out_pivot = out_pivot[[c for c in col_order if c in out_pivot.columns]]
sns.heatmap(
    out_pivot, ax=axes[0], annot=True, fmt=".0f",
    cmap="YlOrRd", linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Outlier months (%)"},
    vmin=0, vmax=30,
)
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_title(
    "Outlier Month Rate (%) — Extreme-Value Records\n"
    "(3 × IQR upper fence  per Category × Facility)"
)
axes[0].tick_params(axis="x", rotation=30)
axes[0].tick_params(axis="y", rotation=0)

# Right — outlier rate vs mean-bias scatter
sc = axes[1].scatter(
    outlier_df["outlier_pct"],
    outlier_df["mean_bias_pct"],
    c=outlier_df["n_outliers"],
    cmap="RdYlGn_r", s=80, alpha=0.85, edgecolors="white",
)
plt.colorbar(sc, ax=axes[1], label="# Outlier Months")
axes[1].axvline(10, color=AMBER, linestyle="--", linewidth=0.9, alpha=0.65,
                label="10% outlier-rate alert")
axes[1].set_xlabel("Outlier Month Rate (%)")
axes[1].set_ylabel("Bias Introduced in Training Mean (%)")
axes[1].set_title(
    "Outlier Impact on Model Training\n"
    "(Top-right → heavily biased training signal for lag-feature ML models)"
)
axes[1].legend(fontsize=8)

fig.suptitle(
    "Outlier Contamination — How Extreme Values Distort Model Training"
    "  (XGBoost / LightGBM / Random Forest most susceptible via lag features)",
    fontsize=11, fontweight="bold", color=BLUE,
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig14_outlier_contamination.pdf", bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 15 — DATA SUFFICIENCY SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 15: Data sufficiency scorecard...")

# Tier thresholds from the modelling notebooks:
#   Tier A  ≥ 24 non-zero months  → full model suite
#   Tier B  12–23 months          → limited models
#   Tier C  < 12 months           → fixed-interval ordering only
def _sufficiency_label(n):
    if   n >= 24: return "Tier A (≥24 mo)"
    elif n >= 12: return "Tier B (12–23 mo)"
    else:         return "Tier C (<12 mo)"

SUFF_ORDER   = ["Tier A (≥24 mo)", "Tier B (12–23 mo)", "Tier C (<12 mo)"]
SUFF_COLOURS = {
    "Tier A (≥24 mo)":   GREEN,
    "Tier B (12–23 mo)": AMBER,
    "Tier C (<12 mo)":   RED,
}

intermittency_df["sufficiency"] = intermittency_df["n_nonzero"].apply(_sufficiency_label)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left — stacked bar by facility
fac_suff = (
    intermittency_df
    .groupby(["branch_short", "sufficiency"])
    .size().unstack(fill_value=0)
    .reindex(columns=SUFF_ORDER, fill_value=0)
)
bottom = np.zeros(len(fac_suff))
for cls in SUFF_ORDER:
    vals = fac_suff[cls].values.astype(float)
    bars = axes[0].bar(
        fac_suff.index, vals, bottom=bottom,
        color=SUFF_COLOURS[cls], edgecolor="white", label=cls,
    )
    for bar, val, bot in zip(bars, vals, bottom):
        if val > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, bot + val / 2,
                str(int(val)), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
            )
    bottom += vals

axes[0].set_ylabel("Number of Category × Facility Pairs")
axes[0].set_title(
    "Data Sufficiency Tier by Facility\n"
    "(Non-zero months in 36-month training window)"
)
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend(fontsize=8, loc="upper right")

# Right — scatter: non-zero months vs zero-inflation rate
sc_cols = [SUFF_COLOURS[s] for s in intermittency_df["sufficiency"]]
axes[1].scatter(
    intermittency_df["n_nonzero"],
    intermittency_df["zero_rate"] * 100,
    c=sc_cols, s=80, alpha=0.85, edgecolors="white",
)
axes[1].axvline(24, color=GREEN, linestyle="--", linewidth=1.1, alpha=0.7,
                label="Tier A threshold (24 mo)")
axes[1].axvline(12, color=AMBER, linestyle="--", linewidth=1.1, alpha=0.7,
                label="Tier B threshold (12 mo)")
axes[1].set_xlabel("Non-Zero Procurement Months (training window)")
axes[1].set_ylabel("Zero-Inflation Rate (%)")
axes[1].set_title(
    "Non-Zero Months vs Zero-Inflation Rate\n"
    "(Bottom-right = data-rich, forecastable series)"
)
_legend_patches = [mpatches.Patch(color=SUFF_COLOURS[c], label=c) for c in SUFF_ORDER]
_legend_lines   = [
    plt.Line2D([0],[0], color=GREEN, linestyle="--", label="Tier A threshold (24 mo)"),
    plt.Line2D([0],[0], color=AMBER, linestyle="--", label="Tier B threshold (12 mo)"),
]
axes[1].legend(handles=_legend_patches + _legend_lines, fontsize=7, loc="upper right")

fig.suptitle(
    "Data Sufficiency Scorecard — Training Data Adequacy per Category × Facility Pair"
    "  (Tier C pairs should not be modelled with ETS / SARIMA / ML)",
    fontsize=11, fontweight="bold", color=BLUE,
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig15_data_sufficiency.pdf", bbox_inches="tight")
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

print("\nModel performance diagnostic summary (training window: Jan 2021 – Dec 2023):")
print(f"  Total category × facility pairs analysed : {len(intermittency_df)}")
print("  Demand class breakdown (Syntetos-Boylan):")
for cls in CLASS_ORDER:
    n   = (intermittency_df["demand_class"] == cls).sum()
    pct = n / len(intermittency_df) * 100
    print(f"    {cls:<14} : {n:2d} pairs  ({pct:.0f}%)")
print("  Data sufficiency tier breakdown:")
for tier in SUFF_ORDER:
    n   = (intermittency_df["sufficiency"] == tier).sum()
    pct = n / len(intermittency_df) * 100
    tier_ascii = tier.replace("\u2265", ">=")
    print(f"    {tier_ascii:<22} : {n:2d} pairs  ({pct:.0f}%)")
avg_zero = intermittency_df["zero_rate"].mean() * 100
print(f"  Avg zero-inflation rate (all pairs) : {avg_zero:.1f}%")
