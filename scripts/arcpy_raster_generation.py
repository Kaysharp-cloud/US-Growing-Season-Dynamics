"""Generate monthly Sen-slope trend TIFFs with ArcPy.


Requires an ArcGIS Python environment with ArcPy and Spatial Analyst available but it is better to run it directly inside Arcgispro.


"""
from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import arcpy
from arcpy.sa import Idw, ExtractByMask
import warnings
warnings.filterwarnings("ignore")


# All configured paths are relative to the project root, one level above scripts.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATION_INFO_FILE = 'data/processed_data/data_intermediate_gs/df_info_needed.csv'
MASK_FILE = 'data/raw_data/FRR_shapefile/farm_resource_regions.shp'
TREND_INPUT_FILES = {
    'PRCP': 'data/processed_data/supplementary/PRCP_monthly_modified_mk.csv',
    'TMAX': 'data/processed_data/supplementary/TMAX_monthly_modified_mk.csv',
    'TMIN': 'data/processed_data/supplementary/TMIN_monthly_modified_mk.csv',
    'Total_Frost_Days': 'data/processed_data/supplementary/Total_Frost_Days_monthly_modified_mk.csv',
    'Wet_Days': 'data/processed_data/supplementary/Wet_Days_monthly_modified_mk.csv',
    'Dry_Days': 'data/processed_data/supplementary/Dry_Days_monthly_modified_mk.csv',
    'GDD': 'data/processed_data/supplementary/GDD_monthly_modified_mk.csv',
    'GDD_wheat': 'data/processed_data/supplementary/GDD_wheat_monthly_modified_mk.csv',
    'CGDD': 'data/processed_data/supplementary/CGDD_monthly_modified_mk.csv',
    'CGDD_wheat': 'data/processed_data/supplementary/CGDD_wheat_monthly_modified_mk.csv',
}
OUTPUT_DIRECTORY = "data/processed_data/Figures_tiff_new"
SCRATCH_DIRECTORY = "data/processed_data/arcpy_scratch"
SCRATCH_GDB_NAME = "scratch.gdb"
SCRATCH_GDB = str(PROJECT_ROOT / SCRATCH_DIRECTORY / SCRATCH_GDB_NAME)

MONTHS = tuple(range(1, 13))
CELL_SIZE = 0.1
IDW_POWER = 2.0
SPATIAL_REFERENCE_EPSG = 4326


def _ensure_spatial_extension():
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True


def _remove_from_arcgis_pro_contents(dataset_paths):
    """
    Removes any layers/tables from CURRENT ArcGIS Pro project that point to dataset_paths.
    Safe to call even outside ArcGIS Pro.
    """
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
    except Exception:
        return  # not running inside ArcGIS Pro

    # Normalize paths (some layers may return slightly different casing)
    targets = set(os.path.normcase(p) for p in dataset_paths if p)

    for m in aprx.listMaps():
        # Remove layers that reference our scratch FC
        for lyr in m.listLayers():
            try:
                if lyr.supports("dataSource"):
                    ds = os.path.normcase(lyr.dataSource)
                    if ds in targets:
                        m.removeLayer(lyr)
            except Exception:
                pass

        # Remove standalone tables that reference our scratch table
        for tbl in m.listTables():
            try:
                if tbl.supports("dataSource"):
                    ds = os.path.normcase(tbl.dataSource)
                    if ds in targets:
                        m.removeTable(tbl)
            except Exception:
                pass


def _df_to_points_fc(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    out_fc: str,
    spatial_ref_epsg: int = 4326,
):
    """
    Writes df -> table (scratch gdb) -> points feature class.
    Returns (out_fc, out_tbl) so caller can delete/cleanup both.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"df must contain '{lat_col}' and '{lon_col}' columns.")

    d = df.copy().dropna(subset=[lat_col, lon_col])
    if d.empty:
        raise ValueError("No valid rows after dropping missing Lat/Lon.")

    clean_cols = {}
    for c in d.columns:
        if c in (lat_col, lon_col):
            clean_cols[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            if pd.api.types.is_numeric_dtype(d[c]):
                clean_cols[c] = pd.to_numeric(d[c], errors="coerce")
            else:
                clean_cols[c] = d[c].astype(str)

    d2 = pd.DataFrame(clean_cols)

    dtype = []
    for c in d2.columns:
        if pd.api.types.is_numeric_dtype(d2[c]):
            dtype.append((c, "f8"))
        else:
            maxlen = int(d2[c].astype(str).str.len().max() or 1)
            maxlen = min(max(maxlen, 1), 255)
            dtype.append((c, f"U{maxlen}"))

    arr = np.empty(len(d2), dtype=dtype)
    for c in d2.columns:
        arr[c] = d2[c].values

    scratch_gdb = SCRATCH_GDB
    table_name = os.path.basename(out_fc) + "_tbl"
    out_tbl = os.path.join(scratch_gdb, table_name)

    if arcpy.Exists(out_tbl):
        arcpy.Delete_management(out_tbl)
    if arcpy.Exists(out_fc):
        arcpy.Delete_management(out_fc)

    arcpy.da.NumPyArrayToTable(arr, out_tbl)

    arcpy.management.XYTableToPoint(
        in_table=out_tbl,
        out_feature_class=out_fc,
        x_field=lon_col,
        y_field=lat_col,
        coordinate_system=arcpy.SpatialReference(spatial_ref_epsg),
    )

    return out_fc, out_tbl


def generate_idw_tiffs_from_df(
    df: pd.DataFrame,
    value_col: str,
    output_dir: str,
    lat_col: str = "Lat",
    lon_col: str = "Lon",
    group_cols=None,
    mask=None,
    cell_size: float = 0.1,
    power: float = 2.0,
    spatial_ref_epsg: int = 4326,
    filename_prefix: str = "",
    cleanup_points: bool = True,            # NEW
    remove_from_contents: bool = True,      # NEW
):
    _ensure_spatial_extension()
    os.makedirs(output_dir, exist_ok=True)

    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in df columns.")

    group_cols = [] if group_cols is None else list(group_cols)
    for gc in group_cols:
        if gc not in df.columns:
            raise ValueError(f"group_col '{gc}' not found in df columns.")

    df_work = df.dropna(subset=[lat_col, lon_col, value_col]).copy()
    if df_work.empty:
        raise ValueError("No valid rows after dropping missing Lat/Lon/value.")

    scratch_gdb = SCRATCH_GDB
    points_fc = os.path.join(scratch_gdb, f"pts_{value_col}")

    points_fc_created = None
    out_tbl_created = None

    try:
        points_fc_created, out_tbl_created = _df_to_points_fc(
            df=df_work,
            lat_col=lat_col,
            lon_col=lon_col,
            out_fc=points_fc,
            spatial_ref_epsg=spatial_ref_epsg,
        )

        def _sql_equals(field, val):
            f = arcpy.AddFieldDelimiters(points_fc_created, field)
            if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
                if pd.isna(val):
                    return None
                return f"{f} = {val}"
            s = str(val).replace("'", "''")
            return f"{f} = '{s}'"

        outputs = []

        if not group_cols:
            lyr_name = f"lyr_{value_col}_all"
            if arcpy.Exists(lyr_name):
                arcpy.management.Delete(lyr_name)

            arcpy.management.MakeFeatureLayer(points_fc_created, lyr_name)

            n = int(arcpy.management.GetCount(lyr_name)[0])
            if n == 0:
                arcpy.management.Delete(lyr_name)
                raise ValueError("No points found in generated feature layer.")

            idw_ras = Idw(lyr_name, value_col, cell_size=cell_size, power=power)
            out_ras = ExtractByMask(idw_ras, mask) if mask is not None else idw_ras

            fname = f"{filename_prefix}{value_col}_IDW.tif" if filename_prefix else f"{value_col}_IDW.tif"
            out_tif = os.path.join(output_dir, fname)
            out_ras.save(out_tif)
            outputs.append(out_tif)

            arcpy.management.Delete(lyr_name)

        else:
            uniq = df_work[group_cols].drop_duplicates()

            for _, row in uniq.iterrows():
                clauses = []
                label_parts = []

                for gc in group_cols:
                    val = row[gc]
                    c = _sql_equals(gc, val)
                    if c is None:
                        clauses = None
                        break
                    clauses.append(c)
                    label_parts.append(f"{gc}-{str(val)}")

                if clauses is None:
                    continue

                where_clause = " AND ".join(clauses)

                lyr_name = "tmp_pts_lyr"
                if arcpy.Exists(lyr_name):
                    arcpy.management.Delete(lyr_name)

                arcpy.management.MakeFeatureLayer(points_fc_created, lyr_name, where_clause)

                n = int(arcpy.management.GetCount(lyr_name)[0])
                if n == 0:
                    arcpy.management.Delete(lyr_name)
                    continue

                idw_ras = Idw(lyr_name, value_col, cell_size=cell_size, power=power)
                out_ras = ExtractByMask(idw_ras, mask) if mask is not None else idw_ras

                safe_label = "_".join(label_parts).replace(" ", "")
                fname = (
                    f"{filename_prefix}{value_col}_{safe_label}_IDW.tif"
                    if filename_prefix else
                    f"{value_col}_{safe_label}_IDW.tif"
                )

                out_tif = os.path.join(output_dir, fname)
                out_ras.save(out_tif)
                outputs.append(out_tif)

                arcpy.management.Delete(lyr_name)

        return outputs

    finally:
        if cleanup_points:
            # Remove from ArcGIS Pro Contents first (so locks are released)
            if remove_from_contents:
                _remove_from_arcgis_pro_contents([points_fc_created, out_tbl_created])

            # Then delete the scratch datasets
            try:
                if out_tbl_created and arcpy.Exists(out_tbl_created):
                    arcpy.Delete_management(out_tbl_created)
            except Exception:
                pass
            try:
                if points_fc_created and arcpy.Exists(points_fc_created):
                    arcpy.Delete_management(points_fc_created)
            except Exception:
                pass


def _project_path(relative_path, setting_name):
    """Resolve a configured relative path within this project."""
    if not relative_path:
        raise ValueError(f"{setting_name} is blank. Set its project-relative input filename.")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{setting_name} must be relative to the project root.")
    resolved = (PROJECT_ROOT / relative).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{setting_name} must stay inside the project folder.")
    return resolved


def _input_file(relative_path, setting_name):
    resolved = _project_path(relative_path, setting_name)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing input for {setting_name}: {relative_path}")
    return resolved


def main():
    # Validate configured inputs before creating or overwriting raster outputs.
    info_path = _input_file(STATION_INFO_FILE, "STATION_INFO_FILE")
    mask_path = _input_file(MASK_FILE, "MASK_FILE")
    trend_paths = {
        var: _input_file(relative, f"TREND_INPUT_FILES[{var!r}]")
        for var, relative in TREND_INPUT_FILES.items()
    }
    for suffix in (".shx", ".dbf", ".prj"):
        if not mask_path.with_suffix(suffix).is_file():
            raise FileNotFoundError(f"Missing mask shapefile component: {mask_path.stem}{suffix}")

    df_info = pd.read_csv(info_path)
    needed_info_cols = {"ID", "Lat", "Lon"}
    missing = needed_info_cols - set(df_info.columns)
    if missing:
        raise ValueError(f"Station metadata is missing columns: {sorted(missing)}")
    df_info = df_info[["ID", "Lat", "Lon"]].drop_duplicates(subset=["ID"]).copy()

    output_dir = _project_path(OUTPUT_DIRECTORY, "OUTPUT_DIRECTORY")
    scratch_dir = _project_path(SCRATCH_DIRECTORY, "SCRATCH_DIRECTORY")
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if not arcpy.Exists(SCRATCH_GDB):
        arcpy.management.CreateFileGDB(str(scratch_dir), SCRATCH_GDB_NAME)

    results = []
    all_outputs = []
    for var, trend_path in trend_paths.items():
        df_trend = pd.read_csv(trend_path)
        needed_trend_cols = {"ID", "MONTH", "SEN_SLOPE"}
        missing = needed_trend_cols - set(df_trend.columns)
        if missing:
            results.append({"var": var, "month": None, "status": "missing_columns", "detail": str(sorted(missing))})
            continue
        df_trend["MONTH"] = pd.to_numeric(df_trend["MONTH"], errors="coerce")
        df_trend_update = df_trend.merge(df_info, on="ID", how="left")
        df_trend_update = df_trend_update.dropna(subset=["Lat", "Lon", "SEN_SLOPE", "MONTH"])
        if df_trend_update.empty:
            results.append({"var": var, "month": None, "status": "no_valid_rows", "detail": ""})
            continue

        for mon in MONTHS:
            df_mon = df_trend_update[df_trend_update["MONTH"] == mon].copy()
            if df_mon.empty:
                results.append({"var": var, "month": int(mon), "status": "skipped_empty", "detail": ""})
                continue
            if df_mon["SEN_SLOPE"].dropna().empty:
                results.append({"var": var, "month": int(mon), "status": "skipped_all_nan", "detail": ""})
                continue
            try:
                outputs = generate_idw_tiffs_from_df(
                    df=df_mon,
                    value_col="SEN_SLOPE",
                    output_dir=str(output_dir),
                    mask=str(mask_path),
                    cell_size=CELL_SIZE,
                    power=IDW_POWER,
                    spatial_ref_epsg=SPATIAL_REFERENCE_EPSG,
                    filename_prefix=f"{var}_Month{int(mon):02d}_trend_",
                    cleanup_points=True,
                    remove_from_contents=True,
                )
                all_outputs.extend(outputs)
                results.append({"var": var, "month": int(mon), "status": "ok", "detail": "; ".join(outputs)})
            except Exception as error:
                results.append({"var": var, "month": int(mon), "status": "failed", "detail": str(error)})

    # Report unsuccessful processing without a stream of progress messages.
    failures = [result for result in results if result["status"] in {"failed", "missing_columns", "no_valid_rows"}]
    if failures:
        raise RuntimeError(f"Raster generation had {len(failures)} unsuccessful jobs. Details: {failures}")
    print(f"Saved {len(all_outputs)} monthly trend TIFFs to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
