"""Growing-season timing, trends, and regional summary CSVs.

Run explicitly with Python; importing this module does not run the analysis.
"""
from pathlib import Path
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pymannkendall as mk
import warnings
warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import data_loader as dl

if Path(dl.__file__).resolve() != (SCRIPTS_DIR / "data_loader.py").resolve():
    raise ImportError("Use the data_loader.py from this script's project directory.")

PROJECT_ROOT = dl.PROJECT_ROOT
INTERMEDIATE_DIR = dl.INTERMEDIATE_DIR
SUPPLEMENTARY_DIR = dl.SUPPLEMENTARY_DIR
FINAL_DIR = dl.FINAL_DIR


def calculate_growing_season_length_with_sos_eos(df, frost_threshold=0):
    """Preserve the original seasonal searches for each station-year.

    SOS: first TMIN > threshold in March-May; if absent, search January-June.
    EOS: first TMIN < threshold in September-November; if absent, search July-December.
    GSL is EOS minus SOS in days. Unresolved dates and lengths remain missing.
    """
    keys = ['ID', 'YEAR']
    data = df[keys + ['DATE', 'TMIN']].copy()
    data['DATE'] = pd.to_datetime(data['DATE'], errors='raise')
    month = data['DATE'].dt.month
    frost_free = data['TMIN'] > frost_threshold
    frost = data['TMIN'] < frost_threshold

    def first_date(condition):
        return data.loc[condition].groupby(keys, observed=True)['DATE'].min()
    sos = first_date(frost_free & month.between(3, 5)).combine_first(first_date(frost_free & month.le(6)))
    eos = first_date(frost & month.between(9, 11)).combine_first(first_date(frost & month.ge(7)))
    result = data[keys].drop_duplicates().sort_values(keys).set_index(keys)
    result = result.join(sos.rename('SOS')).join(eos.rename('EOS')).reset_index()
    result['Growing_Season_Length'] = (result['EOS'] - result['SOS']).dt.days
    result['SOS_num'] = result['SOS'].dt.dayofyear
    result['EOS_num'] = result['EOS'].dt.dayofyear
    return result[['ID', 'YEAR', 'Growing_Season_Length', 'SOS', 'EOS', 'SOS_num', 'EOS_num']]

def fulltrend_mmk_special(df, column_needed, column_name='100years_spei12_', sort_col='DATE', start_year=1994):
    """
    Performs Mann-Kendall trend analysis with Sen's slope estimation on a given column for each unique ID.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_needed (str): The column to analyze trends for.
        column_name (str): Prefix for naming trend columns.
        sort_col (str): Column to sort by before applying trend analysis.
        start_year (int): The cutoff year for filtering data.

    Returns:
        pd.DataFrame: DataFrame containing trend results for each ID.
    """
    df_copy = df.copy()
    df_copy = df_copy.dropna(subset=[column_needed])
    list_trend = []
    for ids in df_copy['ID'].unique():
        df_each = df_copy[df_copy['ID'] == ids].copy()
        df_each = df_each.sort_values(by=sort_col)
        trend_series = df_each[column_needed].values
        if len(trend_series) > 1 and len(set(trend_series)) > 1:
            try:
                result = mk.hamed_rao_modification_test(trend_series)
                sen_result = mk.sens_slope(trend_series)
                trend_data = {'ID': ids, f'{column_name}trend': result.trend, f'{column_name}p': result.p, f'{column_name}slope': sen_result.slope}
                list_trend.append(trend_data)
            except Exception as e:
                pass
    return pd.DataFrame(list_trend, columns=['ID', f'{column_name}trend', f'{column_name}p', f'{column_name}slope'])

def weighted_mean_std(x, w):
    x = pd.to_numeric(x, errors='coerce')
    w = pd.to_numeric(w, errors='coerce')
    m = x.notna() & w.notna() & (w > 0)
    x = x[m].to_numpy(dtype=float)
    w = w[m].to_numpy(dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    mu = np.average(x, weights=w)
    var = np.average((x - mu) ** 2, weights=w)
    return (mu, np.sqrt(var))

def summarize_no_month(df_in, df_basic_info, trend_col, slope_col, domain_col='Name', area_col='Area', conus_label='CONUS'):
    """
    Returns a summary table where rows are domains + CONUS and metrics, columns are metrics (no month).
    """
    df = df_in.merge(df_basic_info[['ID', area_col, domain_col]], on='ID', how='left').copy()
    df[area_col] = pd.to_numeric(df[area_col], errors='coerce')
    df[slope_col] = pd.to_numeric(df[slope_col], errors='coerce')
    conus_area = float(df_basic_info[area_col].sum())
    domain_area = df_basic_info.groupby(domain_col)[area_col].sum().to_dict()

    def compute_block(d, total_area, label):
        inc = d[d[trend_col] == 'increasing']
        dec = d[d[trend_col] == 'decreasing']
        nt = d[d[trend_col] == 'no trend']
        out = {}
        out[f'{label}_PA_increasing'] = inc[area_col].sum() / total_area * 100.0 if total_area > 0 else np.nan
        out[f'{label}_PA_decreasing'] = dec[area_col].sum() / total_area * 100.0 if total_area > 0 else np.nan
        out[f'{label}_PA_notrend'] = nt[area_col].sum() / total_area * 100.0 if total_area > 0 else np.nan
        mu_inc, sd_inc = weighted_mean_std(inc[slope_col], inc[area_col])
        mu_dec, sd_dec = weighted_mean_std(dec[slope_col], dec[area_col])
        out[f'{label}_incr_mn_slope'] = mu_inc
        out[f'{label}_incr_std_slope'] = sd_inc
        out[f'{label}_incr_min_slope'] = pd.to_numeric(inc[slope_col], errors='coerce').min()
        out[f'{label}_incr_max_slope'] = pd.to_numeric(inc[slope_col], errors='coerce').max()
        out[f'{label}_decr_mn_slope'] = mu_dec
        out[f'{label}_decr_std_slope'] = sd_dec
        out[f'{label}_decr_min_slope'] = pd.to_numeric(dec[slope_col], errors='coerce').min()
        out[f'{label}_decr_max_slope'] = pd.to_numeric(dec[slope_col], errors='coerce').max()
        return out
    rows = []
    rows.append(compute_block(df, conus_area, conus_label))
    for name in sorted(df_basic_info[domain_col].dropna().unique()):
        dfi = df[df[domain_col] == name]
        rows.append(compute_block(dfi, float(domain_area.get(name, np.nan)), name))
    summary = pd.DataFrame(rows).T
    summary.columns = [conus_label] + sorted(df_basic_info[domain_col].dropna().unique())
    pa_mask = summary.index.to_series().str.contains('_PA_')
    summary.loc[pa_mask] = summary.loc[pa_mask].round(1)
    summary.loc[~pa_mask] = summary.loc[~pa_mask].round(4)
    return summary

def summarize_avg_by_domain_no_area(df_avg, df_basic_info, value_col, out_csv, domain_col='Name'):
    df = df_avg[['ID', value_col]].merge(df_basic_info[['ID', domain_col]], on='ID', how='left').copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    def stats_row(v, label):
        v = pd.to_numeric(v, errors='coerce').dropna()
        return {'Name': label, 'Min': float(v.min()) if len(v) else np.nan, 'Max': float(v.max()) if len(v) else np.nan, 'Mean': float(v.mean()) if len(v) else np.nan, 'Std': float(v.std(ddof=1)) if len(v) > 1 else np.nan}
    rows = []
    rows.append(stats_row(df[value_col], 'CONUS'))
    for name in sorted(df_basic_info[domain_col].dropna().unique()):
        rows.append(stats_row(df.loc[df[domain_col] == name, value_col], name))
    out = pd.DataFrame(rows)
    out[['Min', 'Max', 'Mean', 'Std']] = out[['Min', 'Max', 'Mean', 'Std']].round(2)
    out.to_csv(out_csv, index=False)
    return out

def tidy_area_trend_summary(path: str, metric: str) -> pd.DataFrame:
    """
    Converts your current 'not presentable' layout (same row/col names with values only at intersections)
    into a clean journal-ready wide table:
      rows = Region
      columns = <metric> + statistic (area % and slope stats)
    """
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: 'row_key'})
    regions = [c for c in df.columns if c != 'row_key']
    stat_order = ['PA_increasing', 'PA_decreasing', 'PA_notrend', 'incr_mn_slope', 'incr_std_slope', 'incr_min_slope', 'incr_max_slope', 'decr_mn_slope', 'decr_std_slope', 'decr_min_slope', 'decr_max_slope']
    data_by_region = {}
    rk = df['row_key'].astype(str)
    for region in regions:
        sub = df[rk.str.startswith(region + '_')].copy()
        d = {}
        for _, r in sub.iterrows():
            stat_key = r['row_key'][len(region) + 1:]
            d[stat_key] = r[region]
        data_by_region[region] = d
    out = pd.DataFrame.from_dict(data_by_region, orient='index')
    out.index.name = 'Region'
    for s in stat_order:
        if s not in out.columns:
            out[s] = np.nan
    out = out[stat_order]
    rename_map = {'PA_increasing': 'Area increasing (%)', 'PA_decreasing': 'Area decreasing (%)', 'PA_notrend': 'Area no trend (%)', 'incr_mn_slope': 'Inc mean slope', 'incr_std_slope': 'Inc slope std', 'incr_min_slope': 'Inc slope min', 'incr_max_slope': 'Inc slope max', 'decr_mn_slope': 'Dec mean slope', 'decr_std_slope': 'Dec slope std', 'decr_min_slope': 'Dec slope min', 'decr_max_slope': 'Dec slope max'}
    out = out.rename(columns=rename_map)
    out = out.add_prefix(f'{metric} ')
    return out

def main():
    for directory in (INTERMEDIATE_DIR, SUPPLEMENTARY_DIR, FINAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    df_daily = dl.load_gsl_data().copy()
    df_info = dl.df_frr_info.copy()
    df_frr_id = dl.df_frr_id.copy()
    df_basic_info = df_frr_id[['ID', 'Area', 'Name']].copy()
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df_daily['SEASON'] = df_daily['MONTH'].map(season_map)
    df_daily['SEASON'] = pd.Categorical(df_daily['SEASON'], categories=['Winter', 'Spring', 'Summer', 'Fall'], ordered=True)
    df_daily.DATE = pd.to_datetime(df_daily.DATE)
    growing_season_df_ext = calculate_growing_season_length_with_sos_eos(df_daily)
    growing_season_df_ext_avg = growing_season_df_ext.groupby(['ID'], observed=True)[['Growing_Season_Length', 'SOS_num', 'EOS_num']].mean().reset_index()
    growing_season_df_ext_avg.to_csv(INTERMEDIATE_DIR / 'gsl_avg.csv')
    df_monthly_updated_needed = df_frr_id[['ID', 'Lat', 'Lon', 'Name']].drop_duplicates()
    growing_season_df_ext_avg = growing_season_df_ext_avg.merge(df_monthly_updated_needed, on='ID', how='left', validate='one_to_one')
    growing_season_df_ext_avg_Eco_region = growing_season_df_ext_avg.groupby('Name')[['Growing_Season_Length', 'SOS_num', 'EOS_num']].agg(['mean', 'min', 'max']).reset_index()
    df_yearly_gs_updated = growing_season_df_ext.copy()
    Growing_Season_Length_trend_ext = fulltrend_mmk_special(df=df_yearly_gs_updated, column_needed='Growing_Season_Length', column_name='Growing_Season_Length', sort_col='YEAR')
    SOS_trend_ext = fulltrend_mmk_special(df=df_yearly_gs_updated, column_needed='SOS_num', column_name='SOS_num', sort_col='YEAR')
    EOS_trend_ext = fulltrend_mmk_special(df=df_yearly_gs_updated, column_needed='EOS_num', column_name='EOS_num', sort_col='YEAR')
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    gsl_summary = summarize_no_month(Growing_Season_Length_trend_ext, df_basic_info, trend_col='Growing_Season_Lengthtrend', slope_col='Growing_Season_Lengthslope')
    gsl_summary.to_csv(SUPPLEMENTARY_DIR / 'Growing_Season_Length_area_trend_summary.csv')
    sos_summary = summarize_no_month(SOS_trend_ext, df_basic_info, trend_col='SOS_numtrend', slope_col='SOS_numslope')
    sos_summary.to_csv(SUPPLEMENTARY_DIR / 'SOS_area_trend_summary.csv')
    eos_summary = summarize_no_month(EOS_trend_ext, df_basic_info, trend_col='EOS_numtrend', slope_col='EOS_numslope')
    eos_summary.to_csv(SUPPLEMENTARY_DIR / 'EOS_area_trend_summary.csv')
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)
    summarize_avg_by_domain_no_area(growing_season_df_ext_avg, df_basic_info, value_col='Growing_Season_Length', out_csv=SUPPLEMENTARY_DIR / 'Growing_Season_Length_avg_summary.csv')
    summarize_avg_by_domain_no_area(growing_season_df_ext_avg, df_basic_info, value_col='SOS_num', out_csv=SUPPLEMENTARY_DIR / 'SOS_num_avg_summary.csv')
    summarize_avg_by_domain_no_area(growing_season_df_ext_avg, df_basic_info, value_col='EOS_num', out_csv=SUPPLEMENTARY_DIR / 'EOS_num_avg_summary.csv')
    in_dir = SUPPLEMENTARY_DIR
    out_dir = FINAL_DIR
    os.makedirs(out_dir, exist_ok=True)
    eos_path = os.path.join(in_dir, 'EOS_area_trend_summary.csv')
    sos_path = os.path.join(in_dir, 'SOS_area_trend_summary.csv')
    gsl_path = os.path.join(in_dir, 'Growing_Season_Length_area_trend_summary.csv')
    eos = tidy_area_trend_summary(eos_path, 'EOS')
    sos = tidy_area_trend_summary(sos_path, 'SOS')
    gsl = tidy_area_trend_summary(gsl_path, 'GSL')
    combined = eos.join(sos, how='outer').join(gsl, how='outer').reset_index()
    area_cols = [c for c in combined.columns if 'Area ' in c]
    slope_cols = [c for c in combined.columns if 'slope' in c.lower()]
    combined[area_cols] = combined[area_cols].round(1)
    combined[slope_cols] = combined[slope_cols].round(4)
    out_csv = os.path.join(out_dir, 'Supplementary_Table_GrowingSeason_EOS_SOS_GSL.csv')
    combined.to_csv(out_csv, index=False)
    combined_t = combined.set_index('Region').T
    combined_t = combined_t.apply(pd.to_numeric, errors='coerce')
    combined_t = combined_t.round(1)
    cols = list(combined_t.columns)
    sorted_cols = (['CONUS'] if 'CONUS' in cols else []) + sorted([c for c in cols if c != 'CONUS'])
    combined_t = combined_t[sorted_cols]
    combined_t_out = combined_t.reset_index().rename(columns={'index': 'Metric/Stat'})
    out_csv_t = os.path.join(out_dir, 'Supplementary_Table_GrowingSeason_EOS_SOS_GSL_TRANSPOSED_1dp.csv')
    combined_t_out.to_csv(out_csv_t, index=False)
    Growing_Season_Length_trend_ext.columns = ['ID', 'TREND', 'P_VALUE', 'SEN_SLOPE']
    Growing_Season_Length_trend_ext.to_csv(INTERMEDIATE_DIR / 'GSL_trend.csv')
    SOS_trend_ext.columns = ['ID', 'TREND', 'P_VALUE', 'SEN_SLOPE']
    SOS_trend_ext.to_csv(INTERMEDIATE_DIR / 'SOS_trend.csv')
    EOS_trend_ext.columns = ['ID', 'TREND', 'P_VALUE', 'SEN_SLOPE']
    EOS_trend_ext.to_csv(INTERMEDIATE_DIR / 'EOS_trend.csv')
    print(f'Saved 12 growing-season CSV files under {dl.PROCESSED_DATA_DIR}')


if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
