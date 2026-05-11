"""
Generate Data Quality and EDA figures for the procurement dataset.
Dataset: data/Clinics Procurement Data.csv  (pipeline run 20260507_083504)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

EHA_BLUE    = '#00548E'
EHA_ACCENT  = '#009976'
EHA_RED     = '#C0392B'
EHA_AMBER   = '#E67E22'
EHA_GREY    = '#F5F5F5'
EHA_LBLUE   = '#E0EDF8'
EHA_PURPLE  = '#8E44AD'

plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('data/Clinics Procurement Data.csv', low_memory=False)
df_raw['date_order'] = pd.to_datetime(df_raw['date_order'])

df_done     = df_raw[df_raw['state'] == 'done'].copy()
df_receipts = df_done[df_done['stock_picking_type'] == 'Receipts'].copy()
df_receipts = df_receipts[df_receipts['product_uom_qty'] > 0].copy()

print(f"Raw records     : {len(df_raw):,}")
print(f"Done records    : {len(df_done):,}")
print(f"Done receipts + qty>0 : {len(df_receipts):,}")
print(f"Date range      : {df_receipts['date_order'].min().date()} to {df_receipts['date_order'].max().date()}")

TARGET_CATS = [
    'Over The Counter Drugs', 'Prescription Medications',
    'Laboratory Consumables', 'Injections', 'Medical Consumables',
    'Vaccines', 'Dental Consumables',
]
TARGET_FACS = ['Asba & Dantata, Abuja', 'Kano - Lamido Crescent', 'Kano - Independence Road']
CAT_LABELS  = ['OTC Drugs', 'Rx Meds', 'Lab Consm.', 'Injections', 'Med. Consm.', 'Vaccines', 'Dental Consm.']
FAC_LABELS  = ['Asba & Dantata\n(Abuja)', 'Kano – Lamido\nCrescent', 'Kano – Independence\nRoad']

# ══════════════════════════════════════════════════════════════════════════════
# DQ FIG 1 — Missing value profile
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
miss     = df_raw.isnull().sum()
miss_pct = (miss / len(df_raw) * 100).sort_values(ascending=True)
miss_pct = miss_pct[miss_pct > 0]
bar_colors = [EHA_RED if v > 50 else EHA_AMBER if v > 10 else EHA_BLUE for v in miss_pct.values]
bars = ax.barh(miss_pct.index, miss_pct.values, color=bar_colors, edgecolor='white', linewidth=0.4, height=0.6)
for bar, v in zip(bars, miss_pct.values):
    ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f'{v:.1f}%', va='center', fontsize=8)
ax.set_xlabel('Missing Values (%)', color=EHA_BLUE)
ax.set_title(f'Missing Value Profile — New Procurement Dataset (n = {len(df_raw):,})', color=EHA_BLUE, fontweight='bold', pad=10)
ax.axvline(10, color=EHA_AMBER, linestyle='--', linewidth=0.8, alpha=0.8)
ax.axvline(50, color=EHA_RED,   linestyle='--', linewidth=0.8, alpha=0.8)
ax.set_xlim(0, 115)
patches = [
    mpatches.Patch(color=EHA_RED,   label='>50% missing (critical)'),
    mpatches.Patch(color=EHA_AMBER, label='10–50% missing (moderate)'),
    mpatches.Patch(color=EHA_BLUE,  label='<10% missing (acceptable)'),
]
ax.legend(handles=patches, fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('reports/figures/dq_missing_values.pdf', bbox_inches='tight')
plt.close()
print('DQ FIG 1: missing value profile — saved')

# ══════════════════════════════════════════════════════════════════════════════
# DQ FIG 2 — Monthly order volume timeline (2018–2026)
# ══════════════════════════════════════════════════════════════════════════════
df_receipts['ym'] = df_receipts['date_order'].dt.to_period('M')
monthly = df_receipts.groupby('ym').size().reset_index(name='count')
monthly['ym_dt'] = monthly['ym'].dt.to_timestamp()

def bar_color_timeline(dt):
    if dt.year < 2022:
        return '#AAAAAA'    # grey: historical, null-branch era
    elif dt.year == 2022 or dt.year == 2023:
        return EHA_AMBER    # amber: partial / ramp-up
    elif dt.year in (2024, 2025):
        return EHA_BLUE     # blue: main training window
    else:
        return EHA_ACCENT   # teal: 2026 validation

bar_colors2 = [bar_color_timeline(r['ym_dt']) for _, r in monthly.iterrows()]

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(monthly['ym_dt'], monthly['count'], color=bar_colors2, width=22, edgecolor='white', linewidth=0.3)
ax.axvline(pd.Timestamp('2024-01-01'), color=EHA_RED,    linestyle='--', linewidth=1.2,
           label='Jan 2024 — Training window start')
ax.axvline(pd.Timestamp('2026-01-01'), color=EHA_ACCENT, linestyle='--', linewidth=1.2,
           label='Jan 2026 — Validation window start')
ax.set_xlabel('Month', color=EHA_BLUE)
ax.set_ylabel('Order Lines (Done Receipts, qty > 0)', color=EHA_BLUE)
ax.set_title('Monthly Order Volume — Full Dataset (Aug 2018 – May 2026)', color=EHA_BLUE, fontweight='bold', pad=10)
patches2 = [
    mpatches.Patch(color='#AAAAAA',  label='2018–2021 (null-branch, pre-reliable)'),
    mpatches.Patch(color=EHA_AMBER,  label='2022–2023 (ramp-up / partial)'),
    mpatches.Patch(color=EHA_BLUE,   label='2024–2025 (training window)'),
    mpatches.Patch(color=EHA_ACCENT, label='2026 (validation window)'),
]
handles2 = patches2 + [
    plt.Line2D([0],[0], color=EHA_RED,    linestyle='--', label='Training start (Jan 2024)'),
    plt.Line2D([0],[0], color=EHA_ACCENT, linestyle='--', label='Validation start (Jan 2026)'),
]
ax.legend(handles=handles2, fontsize=7.5, ncol=2)
plt.tight_layout()
plt.savefig('reports/figures/dq_records_timeline.pdf', bbox_inches='tight')
plt.close()
print('DQ FIG 2: monthly timeline — saved')

# ══════════════════════════════════════════════════════════════════════════════
# DQ FIG 3 — Facility coverage vs original pilot scope
# ══════════════════════════════════════════════════════════════════════════════
new_fac_counts = df_receipts['requesting_branch'].value_counts(dropna=False)

scope_labels = [
    'Asba & Dantata\n(Abuja)',
    'Kano – Lamido\nCrescent',
    'Kano – Independence\nRoad',
    'Abuja – Lugbe',
    'Sangotedo Ajah\n(Lagos)',
    'REACH\n(Kuje)',
]
scope_keys = [
    'Asba & Dantata, Abuja',
    'Kano - Lamido Crescent',
    'Kano - Independence Road',
    'Abuja - Lugbe',
    None,  # Lagos absent
    None,  # REACH absent
]
scope_counts_new = [int(new_fac_counts.get(k, 0)) if k else 0 for k in scope_keys]
bar_colors3 = [EHA_BLUE if v > 500 else EHA_AMBER if v > 0 else EHA_RED for v in scope_counts_new]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

bars3 = ax1.barh(scope_labels, scope_counts_new, color=bar_colors3, edgecolor='white', height=0.55)
for bar, v in zip(bars3, scope_counts_new):
    label = f'{v:,}' if v > 0 else 'ABSENT'
    x_pos = v + 200 if v > 0 else 200
    ax1.text(x_pos, bar.get_y() + bar.get_height() / 2, label, va='center', fontsize=8,
             color=EHA_RED if v == 0 else '#333')
ax1.set_xlabel('Clean Records (done + Receipts + qty > 0)', color=EHA_BLUE)
ax1.set_title('Original 6-Facility Pilot Scope\nvs. New Data Availability', color=EHA_BLUE, fontweight='bold')
ax1.set_xlim(0, 19000)
patches3 = [
    mpatches.Patch(color=EHA_BLUE,  label='Sufficient (>500 records)'),
    mpatches.Patch(color=EHA_AMBER, label='Very sparse (<500 records)'),
    mpatches.Patch(color=EHA_RED,   label='ABSENT from dataset'),
]
ax1.legend(handles=patches3, fontsize=7.5)

# Right panel: all facilities + null branch
all_facs_df = df_receipts['requesting_branch'].value_counts(dropna=False).reset_index()
all_facs_df.columns = ['facility', 'count']
all_facs_df['facility'] = all_facs_df['facility'].fillna('(Null — pre-2022)')
bar_colors4 = [EHA_RED if f == '(Null — pre-2022)' else
               EHA_BLUE if v > 2000 else
               EHA_AMBER if v > 100 else '#AAAAAA'
               for f, v in zip(all_facs_df['facility'], all_facs_df['count'])]
ax2.barh(all_facs_df['facility'], all_facs_df['count'], color=bar_colors4, edgecolor='white', height=0.55)
for i, (_, row) in enumerate(all_facs_df.iterrows()):
    ax2.text(row['count'] + 80, i, f"{row['count']:,}", va='center', fontsize=8)
ax2.set_xlabel('Clean Records (done + Receipts + qty > 0)', color=EHA_BLUE)
ax2.set_title('All Facilities in Dataset\n(incl. null-branch historical records)', color=EHA_BLUE, fontweight='bold')
ax2.set_xlim(0, 18500)

plt.suptitle('Facility Coverage: Original Scope vs. Dataset Reality', color=EHA_BLUE,
             fontweight='bold', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/dq_facility_coverage.pdf', bbox_inches='tight')
plt.close()
print('DQ FIG 3: facility coverage — saved')

# ══════════════════════════════════════════════════════════════════════════════
# DQ FIG 4 — Issues breakdown (state, picking type, flags)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 4.5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.38)

# Panel 1: State pie
ax_s = fig.add_subplot(gs[0, 0])
state_c = df_raw['state'].value_counts()
ax_s.pie(state_c.values,
         labels=[f'{l}\n({v:,})' for l, v in zip(state_c.index, state_c.values)],
         colors=[EHA_BLUE, EHA_AMBER], autopct='%1.1f%%', startangle=90,
         textprops={'fontsize': 9}, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
ax_s.set_title('Order State', color=EHA_BLUE, fontweight='bold')

# Panel 2: Picking type pie
ax_p = fig.add_subplot(gs[0, 1])
spt = df_raw['stock_picking_type'].value_counts()
n_colors = len(spt)
pie_colors = [EHA_BLUE, EHA_ACCENT, EHA_AMBER, '#AAAAAA', '#D5DBDB'][:n_colors]
ax_p.pie(spt.values,
         labels=[f'{l}\n({v:,})' for l, v in zip(spt.index, spt.values)],
         colors=pie_colors,
         autopct='%1.1f%%', startangle=90,
         textprops={'fontsize': 7.5}, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
ax_p.set_title('Stock Picking Type', color=EHA_BLUE, fontweight='bold')

# Panel 3: Flags bar
ax_f = fig.add_subplot(gs[0, 2])
done_receipts_all = df_raw[(df_raw['state'] == 'done') & (df_raw['stock_picking_type'] == 'Receipts')]
n_null_branch = int(df_raw['requesting_branch'].isna().sum())
n_zero_qty    = int((done_receipts_all['product_uom_qty'] == 0).sum())
n_neg_qty     = int((done_receipts_all['product_uom_qty'] < 0).sum())
n_pending  = int((df_raw['state'] == 'purchase').sum())
n_non_receipt = int((df_raw['stock_picking_type'] != 'Receipts').sum())
n_neg_price   = int((df_raw['price_unit'] < 0).sum())

flags = {
    'Null\nbranch': n_null_branch,
    'Zero\nquantity': n_zero_qty,
    'Negative\nquantity': n_neg_qty,
    'Pending\norders': n_pending,
    'Non-receipt\ntransactions': n_non_receipt,
    'Negative\nprice': n_neg_price,
}
flag_colors = [EHA_AMBER, EHA_AMBER, EHA_RED, '#AAAAAA', '#AAAAAA', EHA_RED]
xpos = range(len(flags))
bars_f = ax_f.bar(xpos, list(flags.values()), color=flag_colors, edgecolor='white', width=0.65)
for bar, v in zip(bars_f, flags.values()):
    ax_f.text(bar.get_x() + bar.get_width() / 2, v + 15, str(v),
              ha='center', fontsize=8, fontweight='bold')
ax_f.set_xticks(list(xpos))
ax_f.set_xticklabels(list(flags.keys()), fontsize=7)
ax_f.set_ylabel('Record Count', color=EHA_BLUE)
ax_f.set_title('Data Quality Flags', color=EHA_BLUE, fontweight='bold')

plt.suptitle('Data Quality Flags — Updated Procurement Dataset', color=EHA_BLUE,
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/dq_issues_breakdown.pdf', bbox_inches='tight')
plt.close()
print('DQ FIG 4: issues breakdown — saved')

# ══════════════════════════════════════════════════════════════════════════════
# DQ FIG 5 — Data sufficiency matrix (facility x category, training window only)
# ══════════════════════════════════════════════════════════════════════════════
df_train = df_receipts[
    df_receipts['product_category'].isin(TARGET_CATS) &
    df_receipts['requesting_branch'].isin(TARGET_FACS) &
    (df_receipts['date_order'] >= '2024-01-01') &
    (df_receipts['date_order'] <= '2025-12-31')
].copy()
df_train['ym'] = df_train['date_order'].dt.to_period('M')

pair_monthly = df_train.groupby(
    ['requesting_branch', 'product_category', 'ym']
).size().reset_index(name='records')

active_months = pair_monthly[pair_monthly['records'] > 0].groupby(
    ['requesting_branch', 'product_category']
).size().reset_index(name='active_months')

pivot_am = active_months.pivot_table(
    index='requesting_branch', columns='product_category',
    values='active_months', fill_value=0
).reindex(index=TARGET_FACS, columns=TARGET_CATS, fill_value=0)

fig, ax = plt.subplots(figsize=(9, 4.5))
cmap_custom = LinearSegmentedColormap.from_list('eha', ['#F5F5F5', '#E0EDF8', '#00548E'])
im = ax.imshow(pivot_am.values, cmap=cmap_custom, aspect='auto', vmin=0, vmax=24)

for i in range(len(TARGET_FACS)):
    for j in range(len(TARGET_CATS)):
        val = int(pivot_am.values[i, j])
        color = 'white' if val > 16 else EHA_BLUE
        status = 'OK' if val >= 18 else 'LOW' if val >= 12 else 'INSUFF.' if val > 0 else 'NONE'
        ax.text(j, i, f'{val}\n{status}', ha='center', va='center', fontsize=8,
                fontweight='bold', color=color)

ax.set_xticks(range(len(TARGET_CATS)))
ax.set_xticklabels(CAT_LABELS, fontsize=8.5)
ax.set_yticks(range(len(TARGET_FACS)))
ax.set_yticklabels(FAC_LABELS, fontsize=8.5)
ax.set_title('Data Sufficiency Matrix: Active Months per Facility–Category Pair\n'
             '(Training Window: Jan 2024 – Dec 2025 | max 24 months)',
             color=EHA_BLUE, fontweight='bold', pad=10)
plt.colorbar(im, ax=ax, label='Active months with orders', shrink=0.8)
plt.tight_layout()
plt.savefig('reports/figures/dq_sufficiency_matrix.pdf', bbox_inches='tight')
plt.close()
print('DQ FIG 5: sufficiency matrix — saved')

# ══════════════════════════════════════════════════════════════════════════════
# DQ FIG 6 — Null requesting_branch timeline (pre-2022 records)
# ══════════════════════════════════════════════════════════════════════════════
null_branch = df_raw[df_raw['requesting_branch'].isna()].copy()
null_branch['date_order'] = pd.to_datetime(null_branch['date_order'])
null_branch['ym'] = null_branch['date_order'].dt.to_period('M')
null_timeline = null_branch.groupby('ym').size().reset_index(name='count')
null_timeline['ym_dt'] = null_timeline['ym'].dt.to_timestamp()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.bar(null_timeline['ym_dt'], null_timeline['count'], width=22, color=EHA_AMBER,
        edgecolor='white', linewidth=0.3)
ax1.set_xlabel('Month', color=EHA_BLUE)
ax1.set_ylabel('Records with Null Requesting Branch', color=EHA_BLUE)
ax1.set_title(f'Monthly Count of Null-Branch Records\n(n = {len(null_branch):,} total, 2018–2021)',
              color=EHA_BLUE, fontweight='bold')

# Category breakdown for null branch
null_cat = null_branch['product_category'].value_counts().head(10)
bar_colors_nc = [EHA_AMBER if cat in TARGET_CATS else '#AAAAAA' for cat in null_cat.index]
ax2.barh(null_cat.index[::-1], null_cat.values[::-1], color=bar_colors_nc[::-1],
         edgecolor='white', height=0.6)
for i, (name, cnt) in enumerate(zip(null_cat.index[::-1], null_cat.values[::-1])):
    ax2.text(cnt + 10, i, f'{cnt:,}', va='center', fontsize=8)
ax2.set_xlabel('Record Count', color=EHA_BLUE)
ax2.set_title('Top Categories in Null-Branch Records\n(amber = clinical, grey = non-clinical)',
              color=EHA_BLUE, fontweight='bold')
ax2.set_xlim(0, null_cat.max() * 1.18)

plt.suptitle('Null Requesting Branch — Pre-2022 Historical Records', color=EHA_BLUE,
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/dq_null_branch.pdf', bbox_inches='tight')
plt.close()
print('DQ FIG 6: null branch timeline — saved')

print('\nAll DQ figures saved successfully.\n')

# ══════════════════════════════════════════════════════════════════════════════
# EDA FIG 1 — Category distribution by facility
# ══════════════════════════════════════════════════════════════════════════════
df_focus = df_receipts[
    df_receipts['product_category'].isin(TARGET_CATS) &
    df_receipts['requesting_branch'].isin(TARGET_FACS)
].copy()
df_focus['ym'] = df_focus['date_order'].dt.to_period('M')

cat_fac = df_focus.groupby(['requesting_branch', 'product_category'])['product_uom_qty'].sum().reset_index()
cat_fac_pivot = cat_fac.pivot_table(
    index='product_category', columns='requesting_branch',
    values='product_uom_qty', fill_value=0
).reindex(columns=TARGET_FACS, fill_value=0).reindex(index=TARGET_CATS, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 5))
x       = np.arange(len(TARGET_CATS))
width   = 0.25
colors5 = [EHA_BLUE, EHA_ACCENT, EHA_AMBER]
fac_short = ['Asba & Dantata', 'Kano – Lamido', 'Kano – Independence']

for i, (fac, col, label) in enumerate(zip(TARGET_FACS, colors5, fac_short)):
    vals = cat_fac_pivot[fac].values
    ax.bar(x + (i - 1) * width, vals, width, label=label, color=col, edgecolor='white', linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(CAT_LABELS, fontsize=8.5)
ax.set_ylabel('Total Units Ordered', color=EHA_BLUE)
ax.set_title('Total Units Ordered by Category and Facility\n(Done Receipts — Aug 2018 – May 2026)',
             color=EHA_BLUE, fontweight='bold', pad=10)
ax.legend(fontsize=9, title='Facility', title_fontsize=9)
plt.tight_layout()
plt.savefig('reports/figures/eda_category_by_facility.pdf', bbox_inches='tight')
plt.close()
print('EDA FIG 1: category by facility — saved')

# ══════════════════════════════════════════════════════════════════════════════
# EDA FIG 2 — Monthly trends for top category-facility pairs (training window)
# ══════════════════════════════════════════════════════════════════════════════
top_pairs = [
    ('Asba & Dantata, Abuja',    'Over The Counter Drugs'),
    ('Asba & Dantata, Abuja',    'Prescription Medications'),
    ('Asba & Dantata, Abuja',    'Laboratory Consumables'),
    ('Kano - Lamido Crescent',   'Over The Counter Drugs'),
    ('Kano - Lamido Crescent',   'Prescription Medications'),
    ('Kano - Independence Road', 'Over The Counter Drugs'),
]
pair_labels = [
    'OTC / Asba', 'Rx / Asba', 'Lab Consm. / Asba',
    'OTC / Lamido', 'Rx / Lamido', 'OTC / Independence',
]
pair_colors = [EHA_BLUE, EHA_ACCENT, EHA_AMBER, '#2980B9', '#27AE60', '#E74C3C']

all_months_range = pd.period_range('2024-01', '2025-12', freq='M')

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
axes = axes.flatten()

for ax, (fac, cat), label, col in zip(axes, top_pairs, pair_labels, pair_colors):
    sub = df_focus[(df_focus['requesting_branch'] == fac) & (df_focus['product_category'] == cat)].copy()
    monthly_q = sub.groupby('ym')['product_uom_qty'].sum().reset_index()
    full_idx   = pd.DataFrame({'ym': all_months_range})
    merged     = full_idx.merge(monthly_q[['ym', 'product_uom_qty']], on='ym', how='left').fillna(0)
    merged['ym_dt'] = merged['ym'].dt.to_timestamp()

    ax.bar(merged['ym_dt'], merged['product_uom_qty'], color=col, alpha=0.8, width=22,
           edgecolor='white', linewidth=0.3)
    ax.plot(merged['ym_dt'], merged['product_uom_qty'], color=col, linewidth=1, marker='o',
            markersize=2.5, alpha=0.9)
    ax.set_title(label, color=EHA_BLUE, fontweight='bold', fontsize=9)
    ax.set_ylabel('Units', fontsize=8)
    ax.tick_params(axis='x', rotation=30, labelsize=7)

    z_frac = (merged['product_uom_qty'] == 0).mean() * 100
    ax.text(0.98, 0.97, f'Zero months: {z_frac:.0f}%', transform=ax.transAxes,
            ha='right', va='top', fontsize=7, color=EHA_RED if z_frac > 30 else '#333')

plt.suptitle('Monthly Demand Trends — Top 6 Category-Facility Pairs\n(Training Window: Jan 2024 – Dec 2025)',
             color=EHA_BLUE, fontweight='bold', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('reports/figures/eda_monthly_trends.pdf', bbox_inches='tight')
plt.close()
print('EDA FIG 2: monthly trends — saved')

# ══════════════════════════════════════════════════════════════════════════════
# EDA FIG 3 — Seasonality heatmap
# ══════════════════════════════════════════════════════════════════════════════
df_train_eda = df_focus[
    (df_focus['date_order'] >= '2024-01-01') &
    (df_focus['date_order'] <= '2025-12-31')
].copy()
df_train_eda['month_num'] = df_train_eda['date_order'].dt.month

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pair_series = {}
for fac, cat in top_pairs:
    sub = df_train_eda[(df_train_eda['requesting_branch'] == fac) & (df_train_eda['product_category'] == cat)]
    by_month = sub.groupby('month_num')['product_uom_qty'].mean().reindex(range(1,13), fill_value=0)
    max_val  = by_month.max()
    norm     = (by_month / max_val * 100).round(1) if max_val > 0 else by_month
    pair_series[f'{cat.split()[0]}/{fac.split()[0][:5]}'] = norm.values

hm_df  = pd.DataFrame(pair_series, index=MONTH_NAMES).T

fig, ax = plt.subplots(figsize=(11, 5))
cmap2 = LinearSegmentedColormap.from_list('eha2', ['#F5F5F5', EHA_LBLUE, EHA_BLUE])
sns.heatmap(hm_df, ax=ax, cmap=cmap2, annot=True, fmt='.0f', linewidths=0.4,
            cbar_kws={'label': 'Relative demand (% of peak month)', 'shrink': 0.8},
            annot_kws={'size': 8})
ax.set_title('Seasonality Heatmap — Average Monthly Demand Indexed to Peak\n(Training Window: Jan 2024 – Dec 2025)',
             color=EHA_BLUE, fontweight='bold', pad=10)
ax.set_xlabel('Month', color=EHA_BLUE)
ax.set_ylabel('Category / Facility', color=EHA_BLUE)
plt.tight_layout()
plt.savefig('reports/figures/eda_seasonality_heatmap.pdf', bbox_inches='tight')
plt.close()
print('EDA FIG 3: seasonality heatmap — saved')

# ══════════════════════════════════════════════════════════════════════════════
# EDA FIG 4 — Volatility scatter
# ══════════════════════════════════════════════════════════════════════════════
vol_data = []
for fac in TARGET_FACS:
    for cat in TARGET_CATS:
        sub = df_train_eda[(df_train_eda['requesting_branch'] == fac) & (df_train_eda['product_category'] == cat)]
        monthly_q = sub.groupby('ym')['product_uom_qty'].sum().reindex(all_months_range, fill_value=0)
        n_total   = len(monthly_q)
        n_nonzero = (monthly_q > 0).sum()
        if n_nonzero < 3:
            continue
        mean_q = monthly_q[monthly_q > 0].mean()
        std_q  = monthly_q[monthly_q > 0].std()
        cv     = (std_q / mean_q * 100) if mean_q > 0 else 0
        zero_f = (monthly_q == 0).sum() / n_total * 100
        vol_data.append({
            'pair': f'{cat.split()[0]}/{fac.split(",")[0][:10]}',
            'fac': fac.split()[0],
            'cv': cv,
            'zero_frac': zero_f,
            'mean_monthly': mean_q,
        })

vol_df = pd.DataFrame(vol_data)
fig, ax = plt.subplots(figsize=(9, 6))
unique_facs = vol_df['fac'].unique()
fac_palette = [EHA_BLUE, EHA_ACCENT, EHA_AMBER]
for i, fac in enumerate(unique_facs):
    sub_v = vol_df[vol_df['fac'] == fac]
    ax.scatter(sub_v['zero_frac'], sub_v['cv'],
               s=sub_v['mean_monthly'] / sub_v['mean_monthly'].max() * 300 + 40,
               color=fac_palette[i % len(fac_palette)], alpha=0.75, edgecolors='white',
               linewidth=0.8, label=fac, zorder=3)
    for _, row in sub_v.iterrows():
        ax.annotate(row['pair'], (row['zero_frac'], row['cv']),
                    textcoords='offset points', xytext=(5, 3), fontsize=6.5, color='#333')

ax.axvline(30, color=EHA_AMBER, linestyle='--', linewidth=0.9, alpha=0.7, label='30% zero threshold')
ax.axhline(100, color=EHA_RED,   linestyle='--', linewidth=0.9, alpha=0.7, label='CV=100% threshold')
ax.set_xlabel('Zero-Demand Months (%)', color=EHA_BLUE)
ax.set_ylabel('Coefficient of Variation — Non-Zero Months (%)', color=EHA_BLUE)
ax.set_title('Demand Volatility and Intermittency by Pair\n'
             '(Training Window Jan 2024 – Dec 2025; bubble size = mean monthly volume)',
             color=EHA_BLUE, fontweight='bold', pad=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('reports/figures/eda_volatility_scatter.pdf', bbox_inches='tight')
plt.close()
print('EDA FIG 4: volatility scatter — saved')

# ══════════════════════════════════════════════════════════════════════════════
# EDA FIG 5 — Annual trends (2024 vs 2025)
# ══════════════════════════════════════════════════════════════════════════════
df_focus['year'] = df_focus['date_order'].dt.year
annual_cat = df_focus[df_focus['year'].isin([2024, 2025])].groupby(
    ['product_category', 'year'])['product_uom_qty'].sum().reset_index()
annual_pivot = annual_cat.pivot_table(
    index='product_category', columns='year', values='product_uom_qty', fill_value=0
).reindex(index=TARGET_CATS, fill_value=0)
annual_pivot['yoy_pct'] = ((annual_pivot[2025] - annual_pivot[2024]) /
                           annual_pivot[2024].replace(0, np.nan) * 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x2 = np.arange(len(TARGET_CATS))
w  = 0.35
ax1.bar(x2 - w/2, annual_pivot[2024].values, w, label='2024', color=EHA_BLUE,   edgecolor='white')
ax1.bar(x2 + w/2, annual_pivot[2025].values, w, label='2025', color=EHA_ACCENT, edgecolor='white')
ax1.set_xticks(x2)
ax1.set_xticklabels(CAT_LABELS, fontsize=8.5, rotation=15, ha='right')
ax1.set_ylabel('Total Units Ordered', color=EHA_BLUE)
ax1.set_title('Annual Demand Volume: 2024 vs. 2025\n(All 3 Primary Facilities Combined)',
              color=EHA_BLUE, fontweight='bold')
ax1.legend(fontsize=9)

yoy  = annual_pivot['yoy_pct'].fillna(0).values
cols = [EHA_ACCENT if v >= 0 else EHA_RED for v in yoy]
ax2.bar(CAT_LABELS, yoy, color=cols, edgecolor='white', width=0.55)
ax2.axhline(0, color='black', linewidth=0.8)
for i, v in enumerate(yoy):
    ax2.text(i, v + (2 if v >= 0 else -5), f'{v:+.0f}%', ha='center', fontsize=8, fontweight='bold')
ax2.set_ylabel('Year-on-Year Change (%)', color=EHA_BLUE)
ax2.set_title('Year-on-Year % Change 2024→2025\n(New Dataset)',
              color=EHA_BLUE, fontweight='bold')
ax2.set_xticks(range(len(CAT_LABELS)))
ax2.set_xticklabels(CAT_LABELS, fontsize=8.5, rotation=15, ha='right')

plt.suptitle('Annual Demand Trends — New Dataset', color=EHA_BLUE, fontweight='bold', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/eda_annual_trends.pdf', bbox_inches='tight')
plt.close()
print('EDA FIG 5: annual trends — saved')

# ══════════════════════════════════════════════════════════════════════════════
# EDA FIG 6 — Supplier concentration
# ══════════════════════════════════════════════════════════════════════════════
top_cats_r = df_receipts[df_receipts['product_category'].isin(TARGET_CATS)]
top_sup    = top_cats_r['supplier'].value_counts().head(12)

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors6 = [EHA_BLUE if i < 3 else EHA_ACCENT if i < 6 else '#AAAAAA' for i in range(len(top_sup))]
ax.barh(top_sup.index[::-1], top_sup.values[::-1], color=bar_colors6[::-1], edgecolor='white', height=0.6)
for i, (name, cnt) in enumerate(zip(top_sup.index[::-1], top_sup.values[::-1])):
    ax.text(cnt + 5, i, f'{cnt:,}', va='center', fontsize=8)
ax.set_xlabel('Number of Order Lines', color=EHA_BLUE)
ax.set_title('Top 12 Suppliers for Clinical Categories\n(Done Receipts — Full Dataset)',
             color=EHA_BLUE, fontweight='bold', pad=10)
patches6 = [
    mpatches.Patch(color=EHA_BLUE,   label='Top 3 suppliers'),
    mpatches.Patch(color=EHA_ACCENT, label='Suppliers 4–6'),
    mpatches.Patch(color='#AAAAAA',  label='Suppliers 7–12'),
]
ax.legend(handles=patches6, fontsize=8)
plt.tight_layout()
plt.savefig('reports/figures/eda_supplier_concentration.pdf', bbox_inches='tight')
plt.close()
print('EDA FIG 6: supplier concentration — saved')

# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('SUMMARY STATISTICS FOR REPORTS')
print('='*60)
print(f'\nTotal raw records              : {len(df_raw):,}')
print(f'Done records                   : {len(df_done):,}')
print(f'Done receipts (qty > 0)        : {len(df_receipts):,}')
print(f'Date range                     : {df_receipts["date_order"].min().date()} – {df_receipts["date_order"].max().date()}')
clinical_clean = df_receipts[df_receipts['product_category'].isin(TARGET_CATS) &
                              df_receipts['requesting_branch'].isin(TARGET_FACS)]
print(f'Clinical + primary facs        : {len(clinical_clean):,}')

print('\n--- COMPLETENESS ---')
for col in ['requesting_branch', 'location_1', 'location_2', 'unit_1', 'unit_2']:
    n = df_raw[col].isna().sum()
    p = n / len(df_raw) * 100
    print(f'  {col:<22}: {n:,} ({p:.1f}%)')

print('\n--- ROW-LEVEL FLAGS ---')
done_rec = df_raw[(df_raw['state'] == 'done') & (df_raw['stock_picking_type'] == 'Receipts')]
print(f'  Zero qty (done receipts)     : {(done_rec["product_uom_qty"]==0).sum():,}')
print(f'  Negative qty (done receipts) : {(done_rec["product_uom_qty"]<0).sum():,}')
print(f'  Null rows                    : {df_raw.isnull().all(axis=1).sum():,}')
print(f'  Duplicate order_line_id      : {df_raw["order_line_id"].duplicated().sum():,}')

print('\n--- KANO INDEPENDENCE BY YEAR (clean clinical) ---')
kano_ind_clin = df_receipts[(df_receipts['requesting_branch'] == 'Kano - Independence Road') &
                              (df_receipts['product_category'].isin(TARGET_CATS))]
print(kano_ind_clin['date_order'].dt.year.value_counts().sort_index())

print('\nAll figures and statistics complete.')
