# EHA Clinics — Procurement Demand Forecasting

> Predictive modelling of health commodity procurement across EHA Clinics facilities in Abuja, Kano, and Lagos.

---

## Overview

This repository contains the data, SQL extraction queries, and exploratory analysis scripts for a demand forecasting initiative across EHA Clinics' network of facilities. The project aims to replace reactive, intuition-driven procurement with data-driven forecasts that reduce stockouts, minimise overstocking, and strengthen supply chain responsiveness for critical health commodities.

The work is structured around a clear KPI:

> **Establish a forecast accuracy baseline (MAPE) by Q2, then achieve MAPE ≤ 20% for critical health commodities across pilot facilities by Q3.**

MAPE (Mean Absolute Percentage Error) is the standard accuracy metric for supply chain demand forecasting. A MAPE of ≤ 20% is the accepted threshold for operationally useful forecasts in health supply chain management.

---

## Background

EHA Clinics operates a network of primary and secondary healthcare facilities across Nigeria, with procurement managed through the **Odoo ERP system** and commodity distribution supported by **LoMIS**. Historically, procurement decisions have been driven by periodic manual reviews rather than statistical demand signals, resulting in inconsistent stock levels across facilities and categories.

This initiative addresses that gap by:

1. Extracting and cleaning historical procurement transaction data from Odoo
2. Performing exploratory data analysis to understand demand patterns, seasonality, and data quality
3. Building and validating forecasting models per commodity category and facility
4. Establishing a repeatable monthly monitoring process to track and improve forecast accuracy over time

---

## Repository Structure

```
clinic-procurement/
│
├── data/                                                        # Raw data exports from Odoo ERP
│   ├── Clinics Procurement Data.csv                            # Main dataset (43,799 records, 2018–2024)
│   └── _SELECT_Purchase_Order_Info_..._202604291600.csv        # Supplementary query result export
│
├── queries/                                                     # SQL extraction scripts
│   ├── Query script.sql                                        # Original Odoo procurement query
│   └── Updated Script.sql                                      # Revised query with consumables grouping
│
├── scripts/                                                     # Python analysis scripts
│   └── eda_analysis.py                                         # EDA script — generates 10 diagnostic figures
│
├── reports/                                                     # Generated outputs (local only, not tracked)
│   ├── figures/                                                 # EDA diagnostic figures (PDF)
│   ├── eda_report.pdf                                          # Full EDA report
│   └── forecasting_scope_concept_note.pdf                      # Forecasting scope concept note
│
└── README.md
```

> **Note:** The `reports/` directory is excluded from version control. Run `scripts/eda_analysis.py` to regenerate all figures locally into `reports/figures/`.

---

## Data Description

The primary dataset is a flat export from the Odoo ERP system containing one row per purchase order line item. It spans **2018 to 2024** across **16 branch locations**.

| Field | Description |
|---|---|
| `order_id` | Unique purchase order identifier |
| `purchase_order_name` | Human-readable PO reference (e.g. PO09410) |
| `date_order` | Timestamp of order creation |
| `state` | Order status — `done` (confirmed & received) or `purchase` (pending) |
| `branch_name` | Facility that raised the order |
| `product_id` | Internal product identifier |
| `product_name` | Full product description |
| `category_name` | Procurement category (e.g. Prescription Medications, Vaccines) |
| `category_group` | Grouped category flag (Consumables vs. Others) |
| `product_qty` | Quantity ordered |
| `price_unit` | Unit price (NGN) |
| `price_subtotal` | Total line value (NGN) |
| `month` | Order month (truncated to first of month) |
| `quarter` | Quarter label (e.g. Q2, 2024) |
| `year` | Order year |

### Data Quality Notes

| Issue | Count | % of Total | Handling |
|---|---|---|---|
| Missing branch name | 5,427 | 12.4% | Excluded from facility models |
| Pending state (`purchase`) | 2,704 | 6.2% | Excluded — use `done` only |
| Zero-quantity records | 935 | 2.1% | Excluded from model training |
| Negative spend records | 153 | 0.3% | Credit notes / discounts; excluded |

After applying all filters, the working analytical dataset contains **8,965 records** across 6 pilot facilities and 9 in-scope commodity categories.

---

## Forecasting Scope

### Pilot Facilities

| Facility | City |
|---|---|
| Abuja — Asba and Dantata | Abuja |
| Abuja — Lugbe | Abuja |
| Kano — Lamido Crescent | Kano |
| Kano — Independence Road | Kano |
| Lagos — Sangotedo Ajah | Lagos |
| REACH Abuja Hub 1 Clinic — Kuje | Abuja |

### In-Scope Commodity Categories

- Prescription Medications
- Over The Counter Drugs
- Vaccines / NPI Vaccines
- Injections
- Medical Consumables
- Laboratory Consumables
- Dental Consumables
- Consumables (General)

### Time Horizon

| Window | Period | Purpose |
|---|---|---|
| Training | Jan 2021 — Dec 2023 | Model fitting (post-COVID stable baseline) |
| Validation | Jan 2024 — Jun 2024 | Q2 MAPE baseline measurement |
| Forecast | Rolling 3-month forward | Operational output |

> **Note on pre-2021 data:** Records from 2018–2020 span the clinic expansion period and the COVID-19 disruption. They are retained in the dataset but excluded from the primary training window to avoid embedding abnormal demand patterns into the models.

### Forecast Granularity

The unit of forecast is **commodity category × facility × month**. SKU-level forecasting is scoped for a subsequent phase once category-level MAPE ≤ 20% is achieved.

---

## Key EDA Findings

The `eda_analysis.py` script produces 10 diagnostic figures covering data quality, temporal trends, facility and category profiles, seasonality, forecast readiness, and demand volatility. The most significant findings are:

1. **2021 Kano Independence Road bulk event** — In September 2021, Kano — Independence Road recorded 258,470 units in a single month (vs. a normal baseline of 700–9,000 units/month), driven by a one-time distribution stock-loading exercise involving 10 bulk OTC and prescription products. This event is excluded from all model training to prevent severely distorting demand estimates.

2. **Asba (Abuja) and Kano Lamido are the anchor facilities** — These two sites account for the majority of in-scope spend (NGN 149.9M and NGN 118.3M respectively) and have the longest continuous procurement histories (53+ active months each). Forecasts here will carry the highest statistical confidence.

3. **Laboratory Consumables carry the highest financial risk per forecast error** — Despite ranking fifth by volume (19,191 units), this category accounts for NGN 83.3M in spend — an average unit cost of ~NGN 4,340, roughly 10× the cost of OTC or prescription drugs. Forecast errors here have disproportionately large financial consequences.

4. **OTC Drugs exhibit Q4 Harmattan seasonality** — A clear October–November demand peak is consistent with Nigeria's dry season, during which respiratory infections, skin conditions, and dust-related complaints increase substantially. This is an explicit seasonal effect that must be modelled.

5. **22 of 54 category-facility pairs are data-insufficient for classical time-series modelling** — Dental Consumables, NPI Vaccines, and most REACH Kuje / Lugbe pairs have fewer than 12 non-zero months in the training window. These pairs require either fixed-interval ordering or data collection improvement before forecasting is viable.

---

## Modelling Approach

A **tiered complexity** strategy is applied: use the simplest model that meets the MAPE ≤ 20% target, escalating only when simpler models demonstrably fall short.

| Tier | Models | Trigger |
|---|---|---|
| 1 — Benchmark | Seasonal Naïve, 12-Month Rolling Average | Always applied first; establishes Q2 baseline |
| 2 — Classical | ETS (Holt-Winters), SARIMA, Theta | Default for data-rich pairs (≥24 months) |
| 3 — Additive | Facebook Prophet (with Nigerian holiday regressors) | When Tier 2 fails to hit target |
| 4 — ML (conditional) | XGBoost, LightGBM, Random Forest | Only if Tiers 1–3 insufficient |
| Deferred | LSTM, Temporal Fusion Transformers | Phase 2 (SKU-level, larger data volume) |

Model selection uses **walk-forward cross-validation** on the training window. The winning model is selected independently per category-facility pair.

---

## Running the EDA Script

### Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

### Usage

```bash
python scripts/eda_analysis.py
```

The script resolves all paths relative to its own location — run it from anywhere in the project and it will find `data/Clinics Procurement Data.csv` and write figures to `reports/figures/` automatically. It generates 10 figures saved as PDFs to a `figures/` subdirectory:

| Figure | Content |
|---|---|
| `fig1_data_quality.pdf` | Data quality issues — counts and % of total |
| `fig2_annual_trend.pdf` | Annual procurement trend with 2021 anomaly annotated |
| `fig3_facility_profile.pdf` | Facility comparison — spend, volume, active months |
| `fig4_category_profile.pdf` | Category comparison — spend vs. volume |
| `fig5_monthly_trends.pdf` | Monthly demand trends for top 4 categories |
| `fig6_readiness_heatmap.pdf` | Forecast readiness — non-zero months per category × facility |
| `fig7_seasonality.pdf` | Average monthly volume by calendar month (seasonality) |
| `fig8_volatility_heatmap.pdf` | Demand volatility — Coefficient of Variation (%) |
| `fig9_top_products.pdf` | Top 20 products by total volume |
| `fig10_kano_anomaly.pdf` | Kano Independence Rd September 2021 bulk event drill-down |

---

## SQL Queries

### `Query script.sql`
The original extraction query joining Odoo's `purchase_order`, `purchase_order_line`, `product_template`, and `product_category` tables. Returns all confirmed and pending purchase orders with branch, product, category, quantity, price, and time dimension fields.

### `Updated Script.sql`
An enhanced version of the above that adds a `category_group` field, grouping Consumables, Medical Consumables, Dental Consumables, and Laboratory Consumables into a single `'Consumables'` flag for higher-level analysis. This version is the recommended query for ongoing data extractions.

---

## Project Roadmap

- [x] Define forecasting scope and modelling approach
- [x] Extract and profile historical procurement data (Odoo ERP)
- [x] Exploratory data analysis — patterns, quality, readiness
- [ ] Construct modelling-ready dataset (monthly aggregation, zero-fill)
- [ ] Fit and validate Tier 1–3 models on Tier A category-facility pairs
- [ ] Compute and report Q2 MAPE baseline
- [ ] Iterate models to achieve MAPE ≤ 20% by Q3
- [ ] Build monthly monitoring pipeline for ongoing accuracy tracking
- [ ] Phase 2: SKU-level forecasting for top therapeutic groups

---

## Organisation

**eHealth Africa** — [https://www.ehealthafrica.org](https://www.ehealthafrica.org)

EHA Clinics is a network of technology-driven primary healthcare facilities operated by eHealth Africa across Nigeria, delivering quality healthcare services through data-informed clinical and operational management.
