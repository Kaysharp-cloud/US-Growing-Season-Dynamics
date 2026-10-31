"""Shared data loading for the growing-season notebooks.

Paths are resolved from this file, so inputs always come from this project.
GSL uses load_gsl_data(); other notebooks can keep using df_daily_updated,
df_monthly_updated, and the other existing dataframe attributes. Those larger
climate datasets are loaded and cached when first requested.
"""
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed_data"
INTERMEDIATE_DIR = PROCESSED_DATA_DIR / "data_intermediate_gs"
SUPPLEMENTARY_DIR = PROCESSED_DATA_DIR / "supplementary"
FINAL_DIR = PROCESSED_DATA_DIR / "supplementary_final"
FIGURES_DIR = PROJECT_ROOT / "figures"
RASTER_DIR = PROCESSED_DATA_DIR / "Figures_tiff_new"
FRR_SHAPE_PATH = RAW_DATA_DIR / "FRR_shapefile" / "farm_resource_regions.shp"
CONUS_SHAPE_PATH = RAW_DATA_DIR / "US_CONUS_Shapefile" / "US_CONUS.shp"

# Station metadata is small and shared by all notebooks.
df_frr_info = pd.read_csv(INTERMEDIATE_DIR / "df_frr_info_area.csv")
df_frr_id = pd.read_csv(INTERMEDIATE_DIR / "frr_id_info.csv")
df_frr_id = df_frr_id[df_frr_id["ID"].isin(df_frr_info["ID"])].copy()
df_all_info = pd.read_csv(RAW_DATA_DIR / "HPC_data" / "df_new_usa_needed_50_states.csv")
df_needed_info = df_all_info.loc[
    df_all_info["ID"].isin(df_frr_info["ID"]), ["ID", "Lat", "Lon"]
].copy()


@lru_cache(maxsize=1)
def load_gsl_data():
    """Read and cache only the daily columns needed for growing-season timing.

    Notebooks should copy the returned dataframe before changing it.
    This avoids calculating unrelated GDD and monthly indices for GSL.
    """
    station_ids = set(df_frr_id["ID"])
    chunks = []
    for chunk in pd.read_csv(
        RAW_DATA_DIR / "HPC_data" / "df_daily.csv",
        usecols=["ID", "DATE", "YEAR", "TMIN"],
        dtype={"ID": "string", "YEAR": "int16", "TMIN": "float64"},
        chunksize=250_000,
    ):
        chunks.append(chunk.loc[chunk["ID"].isin(station_ids)])
    daily = pd.concat(chunks, ignore_index=True)
    if daily.empty:
        raise ValueError("No daily observations matched the project station metadata.")
    daily["ID"] = daily["ID"].astype("category")
    daily["DATE"] = pd.to_datetime(daily["DATE"], errors="raise")
    daily["MONTH"] = daily["DATE"].dt.month
    return daily


def calculate_gdd_corn(df_daily):
    """
    Calculate Growing Degree Days for corn (base 10°C, cap at 30°C).

    Parameters:
        df_daily (pd.DataFrame): Daily weather data

    Returns:
        pd.DataFrame: Daily data with GDD column
    """
    df_daily['TMAX_capped30'] = df_daily['TMAX'].apply(lambda x: min(x, 30))
    df_daily['GDD'] = ((df_daily['TMIN'] + df_daily['TMAX_capped30']) / 2) - 10
    df_daily['GDD'] = df_daily['GDD'].apply(lambda x: max(x, 0))

    return df_daily

def calculate_gdd_wheat(df_daily):
    """
    Calculate Growing Degree Days for wheat (base 0°C, cap at 35°C).

    Parameters:
        df_daily (pd.DataFrame): Daily weather data

    Returns:
        pd.DataFrame: Daily data with GDD_wheat column
    """
    df_daily['TMAX_capped35'] = df_daily['TMAX'].apply(lambda x: min(x, 35))
    df_daily['GDD_wheat'] = (
        (df_daily['TMIN'] + df_daily['TMAX_capped35']) / 2
    ) - 0
    df_daily['GDD_wheat'] = df_daily['GDD_wheat'].apply(lambda x: max(x, 0))

    return df_daily

def aggregate_monthly_gdd(df_daily):
    """
    Aggregate daily GDD values to monthly totals.

    Parameters:
        df_daily (pd.DataFrame): Daily weather data with GDD columns

    Returns:
        pd.DataFrame: Monthly GDD aggregations
    """
    df_monthly_gdd = (
        df_daily.groupby(['ID', 'YEAR', 'MONTH'])[['GDD', "GDD_wheat"]]
        .sum()
        .reset_index()
    )

    return df_monthly_gdd

def calculate_frost_days(df_daily):
    """
    Calculate monthly frost days (TMIN <= 0°C).

    Parameters:
        df_daily (pd.DataFrame): Daily weather data

    Returns:
        pd.DataFrame: Monthly frost day counts
    """
    df_frost = (
        df_daily[df_daily['TMIN'] <= 0]
        .groupby(['ID', 'YEAR', 'MONTH'])
        .size()
        .reset_index(name='Total_Frost_Days')
    )

    return df_frost

def calculate_wet_days(df_daily):
    """
    Calculate monthly wet days (PRCP >= 1 mm).

    Parameters:
        df_daily (pd.DataFrame): Daily weather data

    Returns:
        pd.DataFrame: Monthly wet day counts
    """
    df_wet = (
        df_daily[df_daily['PRCP'] >= 1]
        .groupby(['ID', 'YEAR', 'MONTH'])
        .size()
        .reset_index(name='Wet_Days')
    )

    return df_wet

def calculate_dry_days(df_daily):
    """
    Calculate monthly dry days (PRCP < 1 mm).

    Parameters:
        df_daily (pd.DataFrame): Daily weather data

    Returns:
        pd.DataFrame: Monthly dry day counts
    """
    df_dry = (
        df_daily[df_daily['PRCP'] < 1]
        .groupby(['ID', 'YEAR', 'MONTH'])
        .size()
        .reset_index(name='Dry_Days')
    )

    return df_dry

def create_monthly_summary(df_daily_raw):
    """
    Create comprehensive monthly summary with all calculated metrics.

    Parameters:
        df_daily (pd.DataFrame): Daily weather data

    Returns:
        pd.DataFrame: Monthly summary with frost, wet/dry days, and GDD
    """
    df_daily = df_daily_raw.copy()
    df_frost = calculate_frost_days(df_daily)
    df_wet = calculate_wet_days(df_daily)
    df_dry = calculate_dry_days(df_daily)

    df_daily = calculate_gdd_corn(df_daily)
    df_daily = calculate_gdd_wheat(df_daily)
    df_gdd = aggregate_monthly_gdd(df_daily)

    df_summary = df_frost.merge(df_wet, on=['ID', 'YEAR', 'MONTH'], how='outer')
    df_summary = df_summary.merge(df_dry, on=['ID', 'YEAR', 'MONTH'], how='outer')
    df_summary = df_summary.merge(df_gdd, on=['ID', 'YEAR', 'MONTH'], how='outer')

    df_summary.fillna(0, inplace=True)

    return df_summary, df_daily

@lru_cache(maxsize=1)
def load_monthly_data():
    """Load monthly climate indices using bounded-size daily input chunks.

    Monthly totals are additive across chunks. This preserves the existing
    formulas while avoiding a full daily dataframe for monthly-only notebooks.
    Copy the returned dataframe before changing it.
    """
    station_ids = set(df_frr_info["ID"])
    summaries = []
    for chunk in pd.read_csv(
        RAW_DATA_DIR / "HPC_data" / "df_daily.csv",
        usecols=["ID", "YEAR", "MONTH", "PRCP", "TMAX", "TMIN"],
        chunksize=250_000,
    ):
        chunk = chunk.loc[chunk["ID"].isin(station_ids)]
        if chunk.empty:
            continue
        summary, _ = create_monthly_summary(chunk)
        summaries.append(summary)
    if not summaries:
        raise ValueError("No daily observations matched the project station metadata.")
    summary = pd.concat(summaries, ignore_index=True).groupby(
        ["ID", "YEAR", "MONTH"], as_index=False, sort=True
    ).sum()
    monthly = pd.read_csv(
        RAW_DATA_DIR / "HPC_data" / "df_monthly.csv",
        usecols=["ID", "YEAR", "MONTH", "PRCP", "TMAX", "TMIN"],
    )
    monthly = monthly.loc[monthly["ID"].isin(station_ids)]
    monthly = monthly.merge(summary, on=["ID", "YEAR", "MONTH"], how="left")
    monthly["CGDD"] = monthly.groupby(["ID", "YEAR"])["GDD"].cumsum()
    monthly["CGDD_wheat"] = monthly.groupby(["ID", "YEAR"])["GDD_wheat"].cumsum()
    return monthly


@lru_cache(maxsize=1)
def load_growing_season_tables():
    """Load station means/trends produced by GSL and the mapped station IDs.

    Run GSL.ipynb first when the underlying climate inputs change.
    """
    names = ("gsl_avg.csv", "GSL_trend.csv", "df_info_needed.csv")
    paths = [INTERMEDIATE_DIR / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing growing-season input files: " + ", ".join(missing))
    return tuple(pd.read_csv(path) for path in paths)


def load_annual_growing_season_data(station_windows):
    """Aggregate PRCP and maize GDD inside each station's fixed FRR window.

    Windows are inclusive day-of-year boundaries (SOS_DOY, EOS_DOY), held
    constant across years. Process chunks to avoid storing all daily records.
    """
    windows = station_windows.set_index("ID")[["SOS_DOY", "EOS_DOY"]]
    if not windows.index.is_unique:
        raise ValueError("Growing-season windows must have one row per station ID.")
    parts = []
    for chunk in pd.read_csv(
        RAW_DATA_DIR / "HPC_data" / "df_daily.csv",
        usecols=["DATE", "ID", "YEAR", "PRCP", "TMAX", "TMIN"],
        chunksize=250_000,
    ):
        chunk = chunk.loc[chunk["ID"].isin(windows.index)].copy()
        chunk["DATE"] = pd.to_datetime(chunk["DATE"], errors="raise")
        day = chunk["DATE"].dt.dayofyear
        start = chunk["ID"].map(windows["SOS_DOY"])
        end = chunk["ID"].map(windows["EOS_DOY"])
        keep = ((day >= start) & (day <= end)).fillna(False)
        chunk = chunk.loc[keep].copy()
        if chunk.empty:
            continue
        chunk = calculate_gdd_corn(chunk)
        parts.append(chunk.groupby(["ID", "YEAR"], as_index=False).agg(
            PRCP_GS=("PRCP", "sum"),
            GDD_GS=("GDD", "sum"),
            GS_days=("DATE", "count"),
        ))
    if not parts:
        raise ValueError("No observations fall within the supplied growing-season windows.")
    return pd.concat(parts, ignore_index=True).groupby(
        ["ID", "YEAR"], as_index=False, sort=True
    ).sum()


@lru_cache(maxsize=None)
def load_trend_data(variable, monthly=False):
    """Read and cache a generated monthly or growing-season trend CSV.

    Copy the returned dataframe before adding columns or clipping slopes.
    """
    if monthly:
        path = SUPPLEMENTARY_DIR / f"{variable}_monthly_modified_mk.csv"
        producer = "GS_all_indices_july31.ipynb"
    else:
        path = INTERMEDIATE_DIR / f"{variable}_trend.csv"
        producer = "GSL.ipynb"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run {producer} first.")
    data = pd.read_csv(path)
    required = {"ID", "TREND", "P_VALUE", "SEN_SLOPE"}
    if monthly:
        required.add("MONTH")
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    keys = ["ID", "MONTH"] if monthly else ["ID"]
    if data.duplicated(keys).any():
        raise ValueError(f"Duplicate trend records in {path.name} for {keys}.")
    return data


@lru_cache(maxsize=None)
def load_monthly_median_summary(variable):
    """Read a station-median regional summary produced by GS_all_indices."""
    path = SUPPLEMENTARY_DIR / f"{variable}_median_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run GS_all_indices_july31.ipynb first.")
    table = pd.read_csv(path, index_col=0)
    table.columns = pd.to_numeric(table.columns, errors="raise").astype(int)
    if set(table.columns) != set(range(1, 13)):
        raise ValueError(f"{path.name} must contain columns for all twelve months.")
    return table.reindex(columns=range(1, 13))


@lru_cache(maxsize=1)
def load_mapped_station_info():
    """Read the station coordinates used for GIS maps."""
    path = INTERMEDIATE_DIR / "df_info_needed.csv"
    table = pd.read_csv(path)
    required = {"ID", "Lat", "Lon"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path.name} must contain {sorted(required)}.")
    if table["ID"].duplicated().any():
        raise ValueError(f"Duplicate station IDs in {path.name}.")
    return table


# Preserve the dataframe interface used by the other copied notebooks.
_CLIMATE_DATA_NAMES = {
    "df_daily", "df_monthly", "df_summary", "df_daily_updated",
    "df_monthly_updated", "df_monthly_with_loc",
}


def _load_climate_data():
    """Initialize the full climate-index dataframes once, on first access."""
    daily = pd.read_csv(
        RAW_DATA_DIR / "HPC_data" / "df_daily.csv",
        usecols=["DATE", "ID", "YEAR", "MONTH", "DAY", "PRCP", "TMAX", "TMIN"],
    )
    monthly = pd.read_csv(
        RAW_DATA_DIR / "HPC_data" / "df_monthly.csv",
        usecols=["ID", "YEAR", "MONTH", "PRCP", "TMAX", "TMIN"],
    )
    summary, daily_updated = create_monthly_summary(daily)
    monthly_updated = monthly.merge(summary, on=["ID", "YEAR", "MONTH"], how="left")
    daily_updated = daily_updated[daily_updated["ID"].isin(df_frr_info["ID"])]
    monthly_updated = monthly_updated[monthly_updated["ID"].isin(df_frr_info["ID"])].copy()
    monthly_updated["CGDD"] = monthly_updated.groupby(["ID", "YEAR"])["GDD"].cumsum()
    monthly_updated["CGDD_wheat"] = monthly_updated.groupby(["ID", "YEAR"])["GDD_wheat"].cumsum()
    globals().update(
        df_daily=daily,
        df_monthly=monthly,
        df_summary=summary,
        df_daily_updated=daily_updated,
        df_monthly_updated=monthly_updated,
        df_monthly_with_loc=monthly_updated.merge(df_needed_info, on="ID", how="left"),
    )


def __getattr__(name):
    if name == "df_monthly_updated":
        return load_monthly_data()
    if name == "df_monthly_with_loc":
        result = load_monthly_data().merge(df_needed_info, on="ID", how="left")
        globals()[name] = result
        return result
    if name in _CLIMATE_DATA_NAMES:
        _load_climate_data()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
