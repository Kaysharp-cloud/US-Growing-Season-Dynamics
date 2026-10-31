"""Regional growing-season GDD, precipitation, and GSL trend figure.


"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pymannkendall as mk
import matplotlib

# Save figures without opening windows or depending on a desktop display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import data_loader as dl

if Path(dl.__file__).resolve() != (SCRIPTS_DIR / "data_loader.py").resolve():
    raise ImportError("Use the data_loader.py from this script's project directory.")

PROJECT_ROOT = dl.PROJECT_ROOT
SECTION37_DIR = dl.PROCESSED_DATA_DIR / "section_37"
FIGURES_DIR = dl.FIGURES_DIR


def main():
    for directory in (SECTION37_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    gsl_avg, gsl_trend, mapped_stations = (table.copy() for table in dl.load_growing_season_tables())
    df_basic_info = dl.df_frr_id[['ID', 'Name']].copy()
    for label, table in [('station metadata', df_basic_info), ('GSL means', gsl_avg), ('GSL trends', gsl_trend)]:
        if table['ID'].duplicated().any():
            raise ValueError(f'Duplicate station IDs in {label}.')
    regional_windows = gsl_avg[['ID', 'SOS_num', 'EOS_num']].merge(df_basic_info, on='ID', how='inner', validate='one_to_one').groupby('Name', as_index=False)[['SOS_num', 'EOS_num']].mean()
    regional_windows['SOS_DOY'] = np.ceil(regional_windows['SOS_num']).astype('Int64')
    regional_windows['EOS_DOY'] = np.floor(regional_windows['EOS_num']).astype('Int64')
    if regional_windows[['SOS_DOY', 'EOS_DOY']].isna().any().any():
        raise ValueError('A region has no valid mean SOS/EOS; review the GSL inputs.')
    if (regional_windows['SOS_DOY'] > regional_windows['EOS_DOY']).any():
        raise ValueError("A region's growing-season start is after its end.")
    station_windows = df_basic_info.merge(regional_windows, on='Name', validate='many_to_one')
    regional_windows.to_csv(SECTION37_DIR / 'FRR_growing_season_windows.csv', index=False)
    df_annual_gs = dl.load_annual_growing_season_data(station_windows)
    df_annual_gs.to_csv(SECTION37_DIR / 'annual_growing_season_PRCP_GDD.csv', index=False)

    def calculate_gs_trend_by_id(df, value_col, prefix):
        """
    Calculate Hamed-Rao modified Mann-Kendall trend
    and Sen's slope for each station ID.
    """
        results = []
        for station_id, group in df.groupby('ID'):
            group = group.sort_values('YEAR').dropna(subset=[value_col])
            values = group[value_col].to_numpy(dtype=float)
            years = group['YEAR'].to_numpy()
            if len(values) > 1 and len(np.unique(values)) > 1:
                try:
                    mmk_result = mk.hamed_rao_modification_test(values)
                    sen_result = mk.sens_slope(values)
                    slope = float(sen_result.slope)
                    result_row = {'ID': station_id, f'{prefix}_trend': mmk_result.trend, f'{prefix}_p': mmk_result.p, f'{prefix}_slope': slope, f'{prefix}_slope_decade': slope * 10, f'{prefix}_significant': mmk_result.p < 0.05, f'{prefix}_n_years': len(values), f'{prefix}_start_year': int(np.min(years)), f'{prefix}_end_year': int(np.max(years))}
                    if 'Name' in group.columns:
                        result_row['Name'] = group['Name'].iloc[0]
                    results.append(result_row)
                except Exception as error:
                    pass
        return pd.DataFrame(results)
    prcp_trends = calculate_gs_trend_by_id(df_annual_gs, 'PRCP_GS', 'PRCP')
    gdd_trends = calculate_gs_trend_by_id(df_annual_gs, 'GDD_GS', 'GDD')
    df_gs_trend_by_id = prcp_trends.merge(gdd_trends, on='ID', how='outer', validate='one_to_one')
    df_gs_trend_by_id = df_gs_trend_by_id.merge(df_basic_info, on='ID', how='left', validate='one_to_one')
    df_gs_trend_by_id.to_csv(SECTION37_DIR / 'station_growing_season_PRCP_GDD_trends.csv', index=False)
    gsl_station = gsl_trend[['ID', 'SEN_SLOPE']].merge(mapped_stations[['ID']], on='ID', how='inner', validate='one_to_one')
    gsl_station['GSLS_slope_decade_raw'] = pd.to_numeric(gsl_station['SEN_SLOPE'], errors='coerce') * 10
    valid = gsl_station['GSLS_slope_decade_raw'].dropna()
    if valid.empty:
        raise ValueError('No valid GSL slopes are available for the final figure.')
    whiskers = boxplot_stats(valid, whis=1.5)[0]
    gsl_station['GSLS_slope_decade'] = gsl_station['GSLS_slope_decade_raw'].clip(lower=float(whiskers['whislo']), upper=float(whiskers['whishi']))
    gsl_station = gsl_station.merge(df_basic_info, on='ID', how='left', validate='one_to_one')
    gsl_station.to_csv(SECTION37_DIR / 'station_GSL_clipped_trends.csv', index=False)
    df_gsl_mean = gsl_station.groupby('Name', as_index=False)['GSLS_slope_decade'].mean()
    df_gsl_mean['GSLS_slope_decade'] = df_gsl_mean['GSLS_slope_decade'].round(1)
    df_climate_mean = df_gs_trend_by_id.groupby('Name', as_index=False)[['PRCP_slope_decade', 'GDD_slope_decade']].mean()
    df_climate_mean[['PRCP_slope_decade', 'GDD_slope_decade']] = df_climate_mean[['PRCP_slope_decade', 'GDD_slope_decade']].round(0)
    df_plot = df_gsl_mean.merge(df_climate_mean, on='Name', how='inner', validate='one_to_one')
    frr_abbreviations = {'Basin and Range': 'BR', 'Eastern uplands': 'EU', 'Fruitful Rim': 'FR', 'Heartland': 'HL', 'Mississippi Portal': 'MP', 'Northern Crescent': 'NC', 'Northern Great Plains': 'NGP', 'Prairie Gateway': 'PG', 'Southern Seaboard': 'SS'}
    df_plot['FRR'] = df_plot['Name'].map(frr_abbreviations)
    if set(df_plot['Name']) != set(frr_abbreviations) or df_plot.isna().any().any():
        raise ValueError('The final figure requires complete data for all nine FRRs.')
    df_plot.to_csv(SECTION37_DIR / 'FRR_GDD_PRCP_GSL_figure_data.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    size_variable = df_plot['GSLS_slope_decade']
    min_bubble = 600
    max_bubble = 2000
    if size_variable.max() != size_variable.min():
        normalized_size = (size_variable - size_variable.min()) / (size_variable.max() - size_variable.min())
    else:
        normalized_size = pd.Series(0.5, index=size_variable.index)
    bubble_size = min_bubble + normalized_size ** 1.4 * (max_bubble - min_bubble)
    scatter = ax.scatter(df_plot['GDD_slope_decade'], df_plot['PRCP_slope_decade'], s=bubble_size, c=df_plot['GSLS_slope_decade'], cmap='viridis', alpha=0.8, edgecolor='black', linewidth=1.3, zorder=3)
    ax.axhline(0, linestyle='--', linewidth=1.3, alpha=0.7, zorder=1)
    if df_plot['GDD_slope_decade'].min() < 0 < df_plot['GDD_slope_decade'].max():
        ax.axvline(0, linestyle='--', linewidth=1.3, alpha=0.7, zorder=1)
    for _, row in df_plot.iterrows():
        ax.annotate(row['FRR'], (row['GDD_slope_decade'], row['PRCP_slope_decade']), ha='center', va='center', fontsize=11, fontweight='bold', zorder=5)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("GSL Sen's slope (days decade$^{-1}$)", fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    ax.set_xlabel("GDD Sen's slope ($^\\circ$C-days decade$^{-1}$)", fontsize=15, fontweight='bold')
    ax.set_ylabel("PRCP Sen's slope (mm decade$^{-1}$)", fontsize=15, fontweight='bold')
    ax.text(0.98, 0.96, 'Increasing GDD\nIncreasing PRCP', transform=ax.transAxes, ha='right', va='top', fontsize=11, alpha=0.65)
    ax.text(0.98, 0.05, 'Increasing GDD\nDecreasing PRCP', transform=ax.transAxes, ha='right', va='bottom', fontsize=11, alpha=0.65)
    gsl_examples = [size_variable.min(), size_variable.median(), size_variable.max()]
    legend_handles = []
    for gsl_value in gsl_examples:
        if size_variable.max() != size_variable.min():
            normalized = (gsl_value - size_variable.min()) / (size_variable.max() - size_variable.min())
        else:
            normalized = 0.5
        size = min_bubble + normalized ** 1.4 * (max_bubble - min_bubble)
        legend_handle = ax.scatter([], [], s=size, facecolors='none', edgecolors='black', linewidth=1)
        legend_handles.append(legend_handle)
    legend_labels = [f'{value:.1f}' for value in gsl_examples]
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'FRR_GDD_PRCP_GSL_slope_bubble_Sept1.png', dpi=500, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved 5 regional growing-season CSV files to {SECTION37_DIR} and the final figure to {FIGURES_DIR}')


if __name__ == "__main__":
    # Preserve existing numerical behavior; unexpected exceptions still propagate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
