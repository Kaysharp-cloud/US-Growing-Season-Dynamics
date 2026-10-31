"""Monthly agroclimatic medians, modified-MK trends, and regional CSVs.


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


def main():
    for directory in (INTERMEDIATE_DIR, SUPPLEMENTARY_DIR, FINAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    df_monthly = dl.load_monthly_data().copy()
    df_info = dl.df_frr_info.copy()
    df_frr_id = dl.df_frr_id.copy()
    df_frr_id = df_frr_id[df_frr_id.ID.isin(df_info.ID.unique())]
    df_basic_info = df_frr_id[['ID', 'Area', 'Name']]
    df_monthly_median = df_monthly.groupby(['ID', 'MONTH']).median().reset_index()
    df_monthly_median.drop(columns='YEAR', inplace=True)
    col_needed = ['PRCP', 'TMAX', 'TMIN', 'Total_Frost_Days', 'Wet_Days', 'Dry_Days', 'GDD', 'GDD_wheat', 'CGDD', 'CGDD_wheat']
    import numpy as np
    import pandas as pd
    var_list_result = []
    for var in col_needed:
        df_temporary = df_monthly_median[['ID', 'MONTH', var]]
        df_final = df_temporary.merge(df_frr_id[['ID', 'Name']], on='ID', how='left')
        df_domain = df_final.groupby(['Name', 'MONTH'], as_index=False)[var].agg(min='min', max='max', mean='mean', std=lambda s: s.std(ddof=1))
        df_conus = df_final.groupby(['MONTH'], as_index=False)[var].agg(min='min', max='max', mean='mean', std=lambda s: s.std(ddof=1))
        df_conus['Name'] = 'CONUS'
        df_stats = pd.concat([df_conus[['Name', 'MONTH', 'min', 'max', 'mean', 'std']], df_domain], ignore_index=True)
        df_long = df_stats.melt(id_vars=['Name', 'MONTH'], value_vars=['min', 'max', 'mean', 'std'], var_name='stat', value_name=var)
        df_long['Name_stat'] = df_long['Name'] + '_' + df_long['stat']
        df_pivot = df_long.pivot(index='Name_stat', columns='MONTH', values=var).reindex(range(1, 13), axis=1)
        conus_rows = ['CONUS_min', 'CONUS_max', 'CONUS_mean', 'CONUS_std']
        remaining = [r for r in df_pivot.index if r not in conus_rows]
        df_pivot = df_pivot.reindex(conus_rows + sorted(remaining))
        if var in ['TMAX', 'TMIN']:
            df_pivot = df_pivot.round(1)
        else:
            df_pivot = df_pivot.round(0).astype('Int64')
        var_list_result.append(df_pivot)
        df_pivot.to_csv(SUPPLEMENTARY_DIR / f'{var}_median_summary.csv')
    import numpy as np
    import pandas as pd
    for var in col_needed:
        df_temporary = df_monthly_median[['ID', 'MONTH', var]]
        df_final = df_temporary.merge(df_frr_id[['ID', 'Name']], on='ID', how='left')
        df_conus = df_final.groupby(['MONTH'], as_index=False)[var].agg(mean='mean', std=lambda s: s.std(ddof=1))
        df_conus['Name'] = 'CONUS'
        df_for_stats = pd.concat([df_conus[['Name', 'MONTH', 'mean', 'std']], df_final.groupby(['Name', 'MONTH'], as_index=False)[var].agg(mean='mean', std=lambda s: s.std(ddof=1))], ignore_index=True)
        df_long = df_for_stats.melt(id_vars=['Name', 'MONTH'], value_vars=['mean', 'std'], var_name='stat', value_name=var)
        df_long['Name_stat'] = df_long['Name'] + '_' + df_long['stat']
        df_pivot = df_long.pivot(index='Name_stat', columns='MONTH', values=var).reindex(range(1, 13), axis=1)
        conus_rows = ['CONUS_mean', 'CONUS_std']
        remaining = [r for r in df_pivot.index if r not in conus_rows]
        df_pivot = df_pivot.reindex(conus_rows + sorted(remaining))
        if var in ['TMAX', 'TMIN']:
            df_pivot = df_pivot.round(1)
        else:
            df_pivot = df_pivot.round(0).astype('Int64')
        df_pivot.to_csv(FINAL_DIR / f'{var}_median_summary.csv')
    import os
    import numpy as np
    import pandas as pd
    import pymannkendall as mk
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)

    def run_modified_mk(series: pd.Series):
        """Return (trend, p, slope) using a modified Mann-Kendall test. Handles short/empty series."""
        s = pd.to_numeric(series, errors='coerce').dropna().values
        if len(s) < 3:
            return ('insufficient_data', np.nan, np.nan)
        if np.all(s == s[0]):
            return ('no trend', 1.0, 0.0)
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                res = mk.hamed_rao_modification_test(s)
            if not np.isfinite(res.p):
                return ('invalid_test', np.nan, float(mk.sens_slope(s).slope))
            return (res.trend, float(res.p), float(res.slope))
        except (ZeroDivisionError, FloatingPointError, ValueError, RuntimeWarning):
            return ('invalid_test', np.nan, float(mk.sens_slope(s).slope))
    df_monthly_sorted = df_monthly.sort_values(['ID', 'MONTH', 'YEAR']).copy()
    for var in col_needed:
        rows = []
        for (station_id, month), g in df_monthly_sorted.groupby(['ID', 'MONTH'], sort=True):
            g = g.sort_values('YEAR')
            trend, pval, slope = run_modified_mk(g[var])
            rows.append({'ID': station_id, 'MONTH': int(month), 'VAR': var, 'N': int(pd.to_numeric(g[var], errors='coerce').dropna().shape[0]), 'TREND': trend, 'P_VALUE': pval, 'SEN_SLOPE': slope})
        df_out = pd.DataFrame(rows).sort_values(['ID', 'MONTH']).reset_index(drop=True)
        out_csv = SUPPLEMENTARY_DIR / f'{var}_monthly_modified_mk.csv'
        df_out.to_csv(out_csv, index=False)
    import os
    import numpy as np
    import pandas as pd
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)

    def weighted_mean_std(x, w):
        """Area-weighted mean and (population) std. Returns (mean, std)."""
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
        x = x[mask]
        w = w[mask]
        if x.size == 0:
            return (np.nan, np.nan)
        mu = np.average(x, weights=w)
        var = np.average((x - mu) ** 2, weights=w)
        return (mu, np.sqrt(var))

    def summarize_domain_monthly(df, total_area, domain_label):
        """
    df: merged dataframe with columns [ID, MONTH, TREND, SEN_SLOPE, Area, Name]
    total_area: scalar total area for the domain (or CONUS)
    Returns: DataFrame indexed by metric rows, columns are months (1..12)
    """
        months = list(range(1, 13))
        rows = {}
        for month in months:
            d = df[df['MONTH'] == month]
            inc = d[d['TREND'] == 'increasing']
            pa_inc = inc['Area'].sum() / total_area * 100.0 if total_area > 0 else np.nan
            mu_inc, sd_inc = weighted_mean_std(inc['SEN_SLOPE'], inc['Area'])
            dec = d[d['TREND'] == 'decreasing']
            pa_dec = dec['Area'].sum() / total_area * 100.0 if total_area > 0 else np.nan
            mu_dec, sd_dec = weighted_mean_std(dec['SEN_SLOPE'], dec['Area'])
            rows.setdefault(f'{domain_label}_PA_increasing', {})[month] = pa_inc
            rows.setdefault(f'{domain_label}_incr_mn_slope', {})[month] = mu_inc
            rows.setdefault(f'{domain_label}_incr_std_slope', {})[month] = sd_inc
            rows.setdefault(f'{domain_label}_PA_decreasing', {})[month] = pa_dec
            rows.setdefault(f'{domain_label}_decr_mn_slope', {})[month] = mu_dec
            rows.setdefault(f'{domain_label}_decr_std_slope', {})[month] = sd_dec
        out = pd.DataFrame(rows).T
        out = out.reindex(columns=months)
        return out

    def build_area_trend_summary_for_var(var, df_basic_info, supplementary_dir=SUPPLEMENTARY_DIR):
        trend_path = os.path.join(supplementary_dir, f'{var}_monthly_modified_mk.csv')
        df_trend = pd.read_csv(trend_path)
        df = df_trend.merge(df_basic_info[['ID', 'Area', 'Name']], on='ID', how='left')
        df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce').astype('Int64')
        df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
        df['SEN_SLOPE'] = pd.to_numeric(df['SEN_SLOPE'], errors='coerce')
        df = df[df['MONTH'].between(1, 12, inclusive='both')].copy()
        domain_area = df_basic_info.groupby('Name')['Area'].sum().to_dict()
        conus_area = float(df_basic_info['Area'].sum())
        parts = []
        parts.append(summarize_domain_monthly(df, conus_area, 'CONUS'))
        for name in sorted(df_basic_info['Name'].dropna().unique()):
            dfi = df[df['Name'] == name]
            total_area = float(domain_area.get(name, np.nan))
            parts.append(summarize_domain_monthly(dfi, total_area, name))
        df_out = pd.concat(parts, axis=0)
        pa_mask = df_out.index.to_series().str.contains('_PA_')
        df_out.loc[pa_mask] = df_out.loc[pa_mask].round(1)
        pa_mask = df_out.index.to_series().str.contains('_PA_')
        df_out.loc[pa_mask] = df_out.loc[pa_mask].round(1)
        slope_mask = ~pa_mask
        df_out.loc[slope_mask] = df_out.loc[slope_mask].round(4)
        return df_out
    for var in col_needed:
        df_summary = build_area_trend_summary_for_var(var, df_basic_info, supplementary_dir=SUPPLEMENTARY_DIR)
        out_csv = FINAL_DIR / f'{var}_monthly_area_trend_summary.csv'
        df_summary.to_csv(out_csv, index=True)
    import os
    import numpy as np
    import pandas as pd
    SUPPLEMENTARY_DIR.mkdir(parents=True, exist_ok=True)

    def weighted_mean_std(x, w):
        """Area-weighted mean and (population) std. Returns (mean, std)."""
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
        x = x[mask]
        w = w[mask]
        if x.size == 0:
            return (np.nan, np.nan)
        mu = np.average(x, weights=w)
        var = np.average((x - mu) ** 2, weights=w)
        return (mu, np.sqrt(var))

    def summarize_domain_monthly(df, total_area, domain_label):
        """
    df: merged dataframe with columns [ID, MONTH, TREND, SEN_SLOPE, Area, Name]
    total_area: scalar total area for the domain (or CONUS)
    Returns: DataFrame indexed by metric rows, columns are months (1..12)
    """
        months = list(range(1, 13))
        rows = {}
        for month in months:
            d = df[df['MONTH'] == month]
            inc = d[d['TREND'] == 'increasing'].copy()
            pa_inc = inc['Area'].sum() / total_area * 100.0 if total_area > 0 else np.nan
            mu_inc, sd_inc = weighted_mean_std(inc['SEN_SLOPE'], inc['Area'])
            inc_slope_min = pd.to_numeric(inc['SEN_SLOPE'], errors='coerce').min()
            inc_slope_max = pd.to_numeric(inc['SEN_SLOPE'], errors='coerce').max()
            dec = d[d['TREND'] == 'decreasing'].copy()
            pa_dec = dec['Area'].sum() / total_area * 100.0 if total_area > 0 else np.nan
            mu_dec, sd_dec = weighted_mean_std(dec['SEN_SLOPE'], dec['Area'])
            dec_slope_min = pd.to_numeric(dec['SEN_SLOPE'], errors='coerce').min()
            dec_slope_max = pd.to_numeric(dec['SEN_SLOPE'], errors='coerce').max()
            rows.setdefault(f'{domain_label}_PA_increasing', {})[month] = pa_inc
            rows.setdefault(f'{domain_label}_incr_mn_slope', {})[month] = mu_inc
            rows.setdefault(f'{domain_label}_incr_std_slope', {})[month] = sd_inc
            rows.setdefault(f'{domain_label}_incr_min_slope', {})[month] = inc_slope_min
            rows.setdefault(f'{domain_label}_incr_max_slope', {})[month] = inc_slope_max
            rows.setdefault(f'{domain_label}_PA_decreasing', {})[month] = pa_dec
            rows.setdefault(f'{domain_label}_decr_mn_slope', {})[month] = mu_dec
            rows.setdefault(f'{domain_label}_decr_std_slope', {})[month] = sd_dec
            rows.setdefault(f'{domain_label}_decr_min_slope', {})[month] = dec_slope_min
            rows.setdefault(f'{domain_label}_decr_max_slope', {})[month] = dec_slope_max
        out = pd.DataFrame(rows).T
        out = out.reindex(columns=months)
        return out

    def build_area_trend_summary_for_var(var, df_basic_info, supplementary_dir=SUPPLEMENTARY_DIR):
        trend_path = os.path.join(supplementary_dir, f'{var}_monthly_modified_mk.csv')
        df_trend = pd.read_csv(trend_path)
        df = df_trend.merge(df_basic_info[['ID', 'Area', 'Name']], on='ID', how='left')
        df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce').astype('Int64')
        df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
        df['SEN_SLOPE'] = pd.to_numeric(df['SEN_SLOPE'], errors='coerce')
        df = df[df['MONTH'].between(1, 12, inclusive='both')].copy()
        domain_area = df_basic_info.groupby('Name')['Area'].sum().to_dict()
        conus_area = float(df_basic_info['Area'].sum())
        parts = []
        parts.append(summarize_domain_monthly(df, conus_area, 'CONUS'))
        for name in sorted(df_basic_info['Name'].dropna().unique()):
            dfi = df[df['Name'] == name]
            total_area = float(domain_area.get(name, np.nan))
            parts.append(summarize_domain_monthly(dfi, total_area, name))
        df_out = pd.concat(parts, axis=0)
        pa_mask = df_out.index.to_series().str.contains('_PA_')
        df_out.loc[pa_mask] = df_out.loc[pa_mask].round(1)
        pa_mask = df_out.index.to_series().str.contains('_PA_')
        df_out.loc[pa_mask] = df_out.loc[pa_mask].round(1)
        slope_mask = ~pa_mask
        df_out.loc[slope_mask] = df_out.loc[slope_mask].round(4)
        return df_out
    for var in col_needed:
        df_summary = build_area_trend_summary_for_var(var, df_basic_info, supplementary_dir=SUPPLEMENTARY_DIR)
        out_csv = SUPPLEMENTARY_DIR / f'{var}_monthly_area_trend_summary.csv'
        df_summary.to_csv(out_csv, index=True)
    print(f'Saved 50 monthly-climate CSV files under {dl.PROCESSED_DATA_DIR}')


if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
