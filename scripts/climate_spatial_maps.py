"""Final monthly climate and growing-season map exports.


"""
from pathlib import Path
import os
import sys
import string
import calendar
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib

# Save final maps without opening plot windows or requiring a desktop display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.cbook import boxplot_stats
from rasterio.mask import mask
from rasterio.plot import show
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
MAP_OUTPUT_DIR = dl.FIGURES_DIR / "Tiff_to_png_Aug16_2026"
FIGURE_DPI = 500


def main():
    MAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variables = ['PRCP', 'TMAX', 'TMIN', 'GDD', 'GDD_wheat', 'CGDD', 'CGDD_wheat']
    var_list_result = {var: dl.load_monthly_median_summary(var).copy() for var in variables}
    monthly_trend_inputs = {var: dl.load_trend_data(var, monthly=True).copy() for var in variables}
    station_locations = dl.load_mapped_station_info().copy()
    required_rasters = [dl.RASTER_DIR / f'Month{month:02d}_{var}_IDW.tif' for var in variables for month in range(1, 13)]
    required_rasters += [dl.RASTER_DIR / f'{var}_Month{month:02d}_trend_SEN_SLOPE_IDW.tif' for var in variables for month in range(1, 13)]
    required_rasters += [dl.RASTER_DIR / f'{var}_trend_SEN_SLOPE_IDW.tif' for var in ('SOS', 'EOS', 'GSL')]
    required_shapes = [path.with_suffix(suffix) for path in (dl.FRR_SHAPE_PATH, dl.CONUS_SHAPE_PATH) for suffix in ('.shp', '.shx', '.dbf', '.prj')]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_rasters + required_shapes if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing project GIS inputs:\n' + '\n'.join(missing))
    for path in required_rasters:
        with rasterio.open(path) as raster:
            if raster.crs is None or raster.count < 1:
                raise ValueError(f'Invalid georeferenced raster: {path.name}')
    import os
    import string
    import calendar
    import numpy as np
    import rasterio
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from rasterio.mask import mask
    from rasterio.plot import show

    def plot_monthly_climate_rasters_updated(base_path, variable='PRCP', frr_shp_path=None, state_shp_path=None, var_min=None, var_max=None, lower_tail_percentile=None, upper_tail_percentile=97.5, remaining_classes=6, cmap='RdYlGn', title=None, unit='', use_month_abbr=True, colorbar_spacing='uniform', save_dir=MAP_OUTPUT_DIR, save_figure=True, dpi=FIGURE_DPI):
        """
    Plot monthly climate rasters using flexible percentile-based
    segmented classes.

    DEFAULT CLASS STRUCTURE
    -----------------------
    lower_tail_percentile = None
    upper_tail_percentile = 97.5
    remaining_classes = 6

    This produces:

        Class 1:   0.00 - 16.25 percentile
        Class 2:  16.25 - 32.50 percentile
        Class 3:  32.50 - 48.75 percentile
        Class 4:  48.75 - 65.00 percentile
        Class 5:  65.00 - 81.25 percentile
        Class 6:  81.25 - 97.50 percentile
        Class 7:  97.50 - 100 percentile

    OTHER POSSIBILITIES
    -------------------

    1. Upper tail only:

        lower_tail_percentile = None
        upper_tail_percentile = 97.5

    2. Lower tail only:

        lower_tail_percentile = 2.5
        upper_tail_percentile = None

    3. Both tails:

        lower_tail_percentile = 2.5
        upper_tail_percentile = 97.5

    4. No special tails:

        lower_tail_percentile = None
        upper_tail_percentile = None

    remaining_classes controls how many classes are created in
    the portion that remains after accounting for the selected
    tail percentile(s).

    Raster values are clipped to var_min and var_max before
    calculating percentile boundaries and before plotting.

    COLORBAR
    --------
    Colorbar tick labels are placed directly at the boundaries
    between color segments. Therefore, the minimum, every class
    boundary, and the maximum are explicitly shown.
    """
        if var_min is None or var_max is None:
            raise ValueError('Both var_min and var_max must be provided.')
        var_min = float(var_min)
        var_max = float(var_max)
        if var_min >= var_max:
            raise ValueError(f'For {variable}, var_min must be smaller than var_max.')
        if lower_tail_percentile is not None:
            lower_tail_percentile = float(lower_tail_percentile)
            if not 0 < lower_tail_percentile < 100:
                raise ValueError('lower_tail_percentile must be between 0 and 100, or None.')
        if upper_tail_percentile is not None:
            upper_tail_percentile = float(upper_tail_percentile)
            if not 0 < upper_tail_percentile < 100:
                raise ValueError('upper_tail_percentile must be between 0 and 100, or None.')
        if lower_tail_percentile is not None and upper_tail_percentile is not None:
            if lower_tail_percentile >= upper_tail_percentile:
                raise ValueError('lower_tail_percentile must be smaller than upper_tail_percentile.')
        if remaining_classes < 1:
            raise ValueError('remaining_classes must be at least 1.')
        variable_prefix = variable.upper()
        raster_prefix = {'GDD_WHEAT': 'GDD_wheat', 'CGDD_WHEAT': 'CGDD_wheat'}.get(variable_prefix, variable_prefix)
        is_temperature = variable_prefix in {'TMAX', 'TMIN'}
        tif_files = [os.path.join(base_path, f'Month{i:02d}_{raster_prefix}_IDW.tif') for i in range(1, 13)]
        FRR_shape = gpd.read_file(frr_shp_path)
        state_shape = gpd.read_file(state_shp_path)
        os.makedirs(save_dir, exist_ok=True)
        saveas = os.path.join(save_dir, f'{variable_prefix}_monthly_avg_segmented_updated.jpg')
        all_data = []
        original_minimums = []
        original_maximums = []
        total_values_below_min = 0
        total_values_above_max = 0
        for tif_path in tif_files:
            if not os.path.exists(tif_path):
                continue
            with rasterio.open(tif_path) as src:
                FRR_shape_proj = FRR_shape.to_crs(src.crs)
                clipped_raster, _ = mask(src, FRR_shape_proj.geometry, crop=True, filled=False)
                raster_values = clipped_raster[0]
                if np.ma.isMaskedArray(raster_values):
                    raster_values = raster_values.compressed()
                else:
                    raster_values = raster_values.ravel()
                raster_values = raster_values.astype(float)
                raster_values = raster_values[np.isfinite(raster_values)]
                raster_values = raster_values[raster_values >= -100]
                if raster_values.size == 0:
                    continue
                original_minimums.append(np.nanmin(raster_values))
                original_maximums.append(np.nanmax(raster_values))
                total_values_below_min += int(np.sum(raster_values < var_min))
                total_values_above_max += int(np.sum(raster_values > var_max))
                raster_values = np.clip(raster_values, var_min, var_max)
                all_data.append(raster_values)
        if not all_data:
            raise FileNotFoundError(f'No valid raster values were found for {variable_prefix}.')
        combined = np.concatenate(all_data)
        original_min = float(np.nanmin(original_minimums))
        original_max = float(np.nanmax(original_maximums))
        if lower_tail_percentile is not None and upper_tail_percentile is not None:
            internal_edges = np.linspace(lower_tail_percentile, upper_tail_percentile, remaining_classes + 1)
            percentile_edges = np.concatenate([[0], internal_edges, [100]])
            percentile_case = 'Both lower and upper tails'
        elif lower_tail_percentile is None and upper_tail_percentile is not None:
            main_edges = np.linspace(0, upper_tail_percentile, remaining_classes + 1)
            percentile_edges = np.concatenate([main_edges, [100]])
            percentile_case = 'Upper tail only'
        elif lower_tail_percentile is not None and upper_tail_percentile is None:
            main_edges = np.linspace(lower_tail_percentile, 100, remaining_classes + 1)
            percentile_edges = np.concatenate([[0], main_edges])
            percentile_case = 'Lower tail only'
        else:
            percentile_edges = np.linspace(0, 100, remaining_classes + 1)
            percentile_case = 'No separate percentile tails'
        percentile_edges = np.unique(percentile_edges)
        boundaries = np.nanpercentile(combined, percentile_edges)
        boundaries = np.asarray(boundaries, dtype=float)
        boundaries[0] = var_min
        boundaries[-1] = var_max
        original_boundaries = boundaries.copy()
        original_percentile_edges = percentile_edges.copy()
        keep_boundary = np.concatenate([[True], np.diff(boundaries) > 0])
        boundaries = boundaries[keep_boundary]
        percentile_edges_retained = original_percentile_edges[keep_boundary]
        if boundaries[-1] < var_max:
            boundaries = np.append(boundaries, var_max)
            percentile_edges_retained = np.append(percentile_edges_retained, 100)
        else:
            percentile_edges_retained[-1] = 100
        percentile_edges_retained[0] = 0
        if len(boundaries) < len(original_boundaries):
            pass
        if len(boundaries) < 2:
            raise ValueError(f'Not enough unique class boundaries were found for {variable_prefix}.')
        number_of_segments = len(boundaries) - 1
        for class_number, (percentile_low, percentile_high, value_low, value_high) in enumerate(zip(percentile_edges_retained[:-1], percentile_edges_retained[1:], boundaries[:-1], boundaries[1:]), start=1):
            pass
        discrete_cmap = plt.get_cmap(cmap, number_of_segments)
        norm = colors.BoundaryNorm(boundaries=boundaries, ncolors=discrete_cmap.N, clip=True)

        def format_class_value(value):
            if is_temperature:
                return f'{value:.1f}'
            return f'{int(round(value))}'
        boundary_labels = [format_class_value(value) for value in boundaries]
        fig, axes = plt.subplots(4, 3, figsize=(20, 18))
        axes = axes.flatten()
        for i, (tif_path, ax) in enumerate(zip(tif_files, axes)):
            if not os.path.exists(tif_path):
                ax.set_title(f'({string.ascii_lowercase[i]}) Missing', fontsize=20, fontweight='bold')
                ax.axis('off')
                continue
            with rasterio.open(tif_path) as src:
                FRR_shape_proj = FRR_shape.to_crs(src.crs)
                state_shape_proj = state_shape.to_crs(src.crs)
                clipped_raster, clipped_transform = mask(src, FRR_shape_proj.geometry, crop=True, filled=False)
                masked_raster = clipped_raster[0].astype(float)
                masked_raster = np.ma.masked_where(masked_raster < -100, masked_raster)
                plot_raster = np.ma.clip(masked_raster, var_min, var_max)
                show(plot_raster, transform=clipped_transform, ax=ax, cmap=discrete_cmap, norm=norm, zorder=1)
                state_shape_proj.boundary.plot(ax=ax, edgecolor='black', linewidth=1, zorder=2)
                FRR_shape_proj.boundary.plot(ax=ax, edgecolor='black', linewidth=2.5, facecolor='none', zorder=3)
                ax.set_title(f'({string.ascii_lowercase[i]})', fontsize=24, loc='left', fontweight='bold')
                if use_month_abbr:
                    month_name = calendar.month_abbr[i + 1]
                    ax.text(0.5, 1.02, month_name, fontsize=24, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)
                ax.axis('off')
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=discrete_cmap)
        scalar_mappable.set_array([])
        cbar_ax = fig.add_axes([0.18, 0.075, 0.64, 0.025])
        cbar = plt.colorbar(scalar_mappable, cax=cbar_ax, orientation='horizontal', boundaries=boundaries, ticks=boundaries, spacing=colorbar_spacing, drawedges=True)
        cbar.ax.set_xticklabels(boundary_labels, rotation=0, ha='center', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(axis='x', which='major', length=7, width=1.5, direction='out')
        cbar.set_label(unit, fontsize=22, fontweight='bold', labelpad=15)
        cbar.outline.set_linewidth(1.5)
        if cbar.dividers is not None:
            cbar.dividers.set_linewidth(1.2)
        if title:
            fig.suptitle(title, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.14, 1, 0.96])
        if save_figure:
            plt.savefig(saveas, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    base_path = dl.RASTER_DIR
    out_dir = MAP_OUTPUT_DIR
    FRR_shape_path = dl.FRR_SHAPE_PATH
    state_shp_path = dl.CONUS_SHAPE_PATH
    variables = ['PRCP', 'TMAX', 'TMIN', 'GDD', 'GDD_wheat', 'CGDD', 'CGDD_wheat']
    var_units = {'PRCP': 'PRCP (mm)', 'TMAX': 'Temperature (°C)', 'TMIN': 'Temperature (°C)', 'GDD': 'GDD (°C)', 'GDD_wheat': 'GDD wheat (°C)', 'CGDD': 'CGDD (°C)', 'CGDD_wheat': 'CGDD wheat (°C)'}
    for var in variables:
        variable_values = np.asarray(var_list_result[var], dtype=float)
        var_min = float(np.nanmin(variable_values))
        var_max = float(np.nanmax(variable_values))
        selected_cmap = 'RdYlGn' if var.upper() == 'PRCP' else 'RdYlGn_r'
        plot_monthly_climate_rasters_updated(base_path=base_path, variable=var, frr_shp_path=FRR_shape_path, state_shp_path=state_shp_path, var_min=var_min, var_max=var_max, lower_tail_percentile=None, upper_tail_percentile=97.5, remaining_classes=6, cmap=selected_cmap, unit=var_units.get(var, ''), colorbar_spacing='uniform', title=None, save_dir=out_dir, save_figure=True, dpi=FIGURE_DPI)
    import os
    import string
    import calendar
    import numpy as np
    import pandas as pd
    import rasterio
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from matplotlib.cbook import boxplot_stats
    from rasterio.mask import mask
    from rasterio.plot import show

    def plot_monthly_climate_rasters_with_points_updated(base_path, variable='PRCP', frr_shp_path=None, state_shp_path=None, var_min=None, var_max=None, lower_tail_percentile=2.5, upper_tail_percentile=97.5, middle_classes=5, scale_factor=10, force_zero_break=True, cmap='RdYlGn', title=None, unit='', point_df=None, point_filter_col='P_VALUE', point_filter_thresh=0.05, use_month_abbr=True, colorbar_spacing='uniform', save_dir=MAP_OUTPUT_DIR, save_figure=True, dpi=FIGURE_DPI):
        """
    Plot monthly Sen's slope rasters using exactly seven
    percentile-based color segments.

    Default percentile structure
    ----------------------------

        Class 1:
            0 to 2.5 percentile

        Classes 2 to 6:
            Five equally spaced percentile classes between
            2.5 and 97.5 percentiles

        Class 7:
            97.5 to 100 percentile

    This always produces:

        8 boundaries
        7 color segments

    ZERO HANDLING
    -------------
    If the range contains both negative and positive trends,
    zero is used as one of the EXISTING class boundaries.

    Zero is NOT inserted as an additional boundary.

    Instead, the internal boundary closest to zero is changed
    to exactly zero.

    Example:

        Original:
            -5, 1, 2

        Updated:
            -5, 0, 2

    Therefore, the number of segments does not increase.

    COLORBAR
    --------
    Tick labels are placed directly at the class boundaries.

    Example:

        -5       0       2       5
         |-------|-------|-------|

    rather than:

        -5 to 0    0 to 2    2 to 5
    """
        if not 0 < lower_tail_percentile < upper_tail_percentile < 100:
            raise ValueError('Percentiles must satisfy: 0 < lower_tail_percentile < upper_tail_percentile < 100.')
        if middle_classes != 5:
            raise ValueError('This updated function is designed to always produce 7 segments. Therefore, middle_classes must equal 5.')
        if scale_factor <= 0:
            raise ValueError('scale_factor must be greater than zero.')
        variable_prefix = variable.upper()
        raster_prefix = {'GDD_WHEAT': 'GDD_wheat', 'CGDD_WHEAT': 'CGDD_wheat'}.get(variable_prefix, variable_prefix)
        is_temperature = variable_prefix in {'TMAX', 'TMIN'}
        tif_files = [os.path.join(base_path, f'{raster_prefix}_Month{i:02d}_trend_SEN_SLOPE_IDW.tif') for i in range(1, 13)]
        FRR_shape = gpd.read_file(frr_shp_path)
        state_shape = gpd.read_file(state_shp_path)
        os.makedirs(save_dir, exist_ok=True)
        saveas = os.path.join(save_dir, f'{variable_prefix}_monthly_trend_segmented_updated.jpg')
        all_data = []
        for tif_path in tif_files:
            if not os.path.exists(tif_path):
                continue
            with rasterio.open(tif_path) as src:
                FRR_shape_proj = FRR_shape.to_crs(src.crs)
                clipped_raster, _ = mask(src, FRR_shape_proj.geometry, crop=True, filled=False)
                raster_values = clipped_raster[0]
                if np.ma.isMaskedArray(raster_values):
                    raster_values = raster_values.compressed()
                else:
                    raster_values = raster_values.ravel()
                raster_values = raster_values.astype(float)
                raster_values = raster_values[np.isfinite(raster_values)]
                raster_values = raster_values[raster_values >= -100]
                if raster_values.size == 0:
                    continue
                raster_values = raster_values * scale_factor
                all_data.append(raster_values)
        if not all_data:
            raise FileNotFoundError(f'No valid raster values were found for {variable_prefix}.')
        combined_original = np.concatenate(all_data)
        actual_min = float(np.nanmin(combined_original))
        actual_max = float(np.nanmax(combined_original))
        applied_min = actual_min if var_min is None else float(var_min)
        applied_max = actual_max if var_max is None else float(var_max)
        if applied_min >= applied_max:
            raise ValueError(f'For {variable_prefix}, var_min must be smaller than var_max.')
        values_below_min = int(np.sum(combined_original < applied_min))
        values_above_max = int(np.sum(combined_original > applied_max))
        combined = np.clip(combined_original, applied_min, applied_max)
        middle_percentile_edges = np.linspace(lower_tail_percentile, upper_tail_percentile, middle_classes + 1)
        percentile_edges = np.concatenate([[0], middle_percentile_edges, [100]])
        boundaries = np.nanpercentile(combined, percentile_edges)
        boundaries = np.asarray(boundaries, dtype=float)
        boundaries[0] = applied_min
        boundaries[-1] = applied_max
        expected_boundary_count = 8
        if len(boundaries) != expected_boundary_count:
            raise RuntimeError(f'Expected 8 boundaries but obtained {len(boundaries)}.')
        if np.any(np.diff(boundaries) <= 0):
            total_range = applied_max - applied_min
            epsilon = max(total_range * 1e-08, 1e-12)
            for boundary_index in range(1, len(boundaries) - 1):
                if boundaries[boundary_index] <= boundaries[boundary_index - 1]:
                    boundaries[boundary_index] = boundaries[boundary_index - 1] + epsilon
            for boundary_index in range(len(boundaries) - 2, 0, -1):
                if boundaries[boundary_index] >= boundaries[boundary_index + 1]:
                    boundaries[boundary_index] = boundaries[boundary_index + 1] - epsilon
            boundaries[0] = applied_min
            boundaries[-1] = applied_max
        if force_zero_break and applied_min < 0 and (applied_max > 0):
            data_range = applied_max - applied_min
            zero_tolerance = max(1e-12, data_range * 1e-09)
            near_zero_mask = np.isclose(boundaries, 0.0, atol=zero_tolerance, rtol=0)
            if np.any(near_zero_mask):
                zero_index = int(np.argmin(np.abs(boundaries)))
                old_boundary = boundaries[zero_index]
                boundaries[zero_index] = 0.0
            else:
                internal_boundaries = boundaries[1:-1]
                closest_internal_index = int(np.argmin(np.abs(internal_boundaries)))
                boundary_index = closest_internal_index + 1
                old_boundary = float(boundaries[boundary_index])
                boundaries[boundary_index] = 0.0
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError(f'Boundary adjustment resulted in non-increasing boundaries for {variable_prefix}.')
        number_of_segments = len(boundaries) - 1
        if number_of_segments != 7:
            raise RuntimeError(f'{variable_prefix} produced {number_of_segments} segments instead of 7.')
        for class_number, (value_low, value_high) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            if value_high <= 0:
                trend_direction = 'negative'
            elif value_low >= 0:
                trend_direction = 'positive'
            else:
                trend_direction = 'mixed'
        discrete_cmap = plt.get_cmap(cmap, number_of_segments)
        norm = colors.BoundaryNorm(boundaries=boundaries, ncolors=discrete_cmap.N, clip=True)

        def format_boundary_value(value):
            if is_temperature:
                if np.isclose(value, 0):
                    value = 0.0
                return f'{value:.1f}'
            value = int(round(value))
            if value == 0:
                value = 0
            return f'{value}'
        boundary_labels = [format_boundary_value(value) for value in boundaries]
        points_available = point_df is not None and point_filter_col is not None and (point_filter_thresh is not None)
        if points_available:
            required_point_columns = {'MONTH', 'Lat', 'Lon', point_filter_col}
            missing_point_columns = required_point_columns - set(point_df.columns)
            if missing_point_columns:
                raise ValueError(f'Point DataFrame for {variable_prefix} is missing: {sorted(missing_point_columns)}')
            point_df = point_df.copy()
            point_df['MONTH'] = pd.to_numeric(point_df['MONTH'], errors='coerce')
            point_df['Lat'] = pd.to_numeric(point_df['Lat'], errors='coerce')
            point_df['Lon'] = pd.to_numeric(point_df['Lon'], errors='coerce')
            point_df[point_filter_col] = pd.to_numeric(point_df[point_filter_col], errors='coerce')
        fig, axes = plt.subplots(4, 3, figsize=(20, 18))
        axes = axes.flatten()
        for i, (tif_path, ax) in enumerate(zip(tif_files, axes)):
            if not os.path.exists(tif_path):
                ax.set_title(f'({string.ascii_lowercase[i]}) Missing', fontsize=20, fontweight='bold')
                ax.axis('off')
                continue
            with rasterio.open(tif_path) as src:
                FRR_shape_proj = FRR_shape.to_crs(src.crs)
                state_shape_proj = state_shape.to_crs(src.crs)
                clipped_raster, clipped_transform = mask(src, FRR_shape_proj.geometry, crop=True, filled=False)
                plot_raster = clipped_raster[0].astype(float)
                plot_raster = np.ma.masked_invalid(plot_raster)
                plot_raster = np.ma.masked_where(plot_raster < -100, plot_raster)
                plot_raster = plot_raster * scale_factor
                plot_raster = np.ma.clip(plot_raster, applied_min, applied_max)
                show(plot_raster, transform=clipped_transform, ax=ax, cmap=discrete_cmap, norm=norm, zorder=1)
                state_shape_proj.boundary.plot(ax=ax, edgecolor='black', linewidth=1, zorder=2)
                FRR_shape_proj.boundary.plot(ax=ax, edgecolor='black', linewidth=2.5, facecolor='none', zorder=3)
                if points_available:
                    df_month = point_df.loc[point_df['MONTH'] == i + 1].copy()
                    df_filtered = df_month.loc[df_month[point_filter_col] < point_filter_thresh].copy()
                    df_filtered = df_filtered.dropna(subset=['Lat', 'Lon', point_filter_col])
                    if not df_filtered.empty:
                        gdf_points = gpd.GeoDataFrame(df_filtered, geometry=gpd.points_from_xy(df_filtered['Lon'], df_filtered['Lat']), crs='EPSG:4326').to_crs(src.crs)
                        ax.scatter(gdf_points.geometry.x, gdf_points.geometry.y, facecolors='none', edgecolors='black', linewidths=1, s=50, alpha=0.99, marker='*', zorder=4)
                ax.set_title(f'({string.ascii_lowercase[i]})', fontsize=24, loc='left', fontweight='bold')
                if use_month_abbr:
                    month_name = calendar.month_abbr[i + 1]
                    ax.text(0.5, 1.02, month_name, fontsize=24, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)
                ax.axis('off')
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=discrete_cmap)
        scalar_mappable.set_array([])
        cbar_ax = fig.add_axes([0.18, 0.075, 0.64, 0.025])
        cbar = plt.colorbar(scalar_mappable, cax=cbar_ax, orientation='horizontal', boundaries=boundaries, ticks=boundaries, spacing=colorbar_spacing, drawedges=True)
        cbar.ax.set_xticklabels(boundary_labels, rotation=0, ha='center', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(axis='x', which='major', length=7, width=1.5, direction='out')
        cbar.set_label(unit, fontsize=24, fontweight='bold', labelpad=22)
        cbar.outline.set_linewidth(1.5)
        if cbar.dividers is not None:
            cbar.dividers.set_linewidth(1.2)
        if title:
            fig.suptitle(title, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.14, 1, 0.96])
        if save_figure:
            plt.savefig(saveas, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    raster_dir = dl.RASTER_DIR
    trend_dir = dl.SUPPLEMENTARY_DIR
    info_path = dl.INTERMEDIATE_DIR / 'df_info_needed.csv'
    figure_dir = MAP_OUTPUT_DIR
    FRR_shape_path = dl.FRR_SHAPE_PATH
    state_shp_path = dl.CONUS_SHAPE_PATH
    os.makedirs(figure_dir, exist_ok=True)
    useful_cols = ['PRCP', 'TMAX', 'TMIN', 'GDD', 'GDD_wheat', 'CGDD', 'CGDD_wheat']
    trend_unit_labels = {'PRCP': '$\\mathrm{mm}\\,\\mathrm{decade}^{-1}$', 'TMAX': '$^{\\circ}\\mathrm{C}\\,\\mathrm{decade}^{-1}$', 'TMIN': '$^{\\circ}\\mathrm{C}\\,\\mathrm{decade}^{-1}$', 'GDD': '$^{\\circ}\\mathrm{C}\\,\\mathrm{day}\\,\\mathrm{decade}^{-1}$', 'GDD_wheat': '$^{\\circ}\\mathrm{C}\\,\\mathrm{day}\\,\\mathrm{decade}^{-1}$', 'CGDD': '$^{\\circ}\\mathrm{C}\\,\\mathrm{day}\\,\\mathrm{decade}^{-1}$', 'CGDD_wheat': '$^{\\circ}\\mathrm{C}\\,\\mathrm{day}\\,\\mathrm{decade}^{-1}$'}
    trend_limits = {'PRCP': (None, None), 'TMAX': (None, None), 'TMIN': (None, None), 'GDD': (None, None), 'GDD_wheat': (None, None), 'CGDD': (None, None), 'CGDD_wheat': (None, None)}
    df_info = station_locations.copy()
    needed_info_cols = {'ID', 'Lat', 'Lon'}
    missing_info_cols = needed_info_cols - set(df_info.columns)
    if missing_info_cols:
        raise ValueError(f'df_info_needed.csv is missing columns: {sorted(missing_info_cols)}')
    df_info = df_info[['ID', 'Lat', 'Lon']].drop_duplicates(subset=['ID']).copy()
    run_results = []
    for var in useful_cols:
        trend_path = os.path.join(trend_dir, f'{var}_monthly_modified_mk.csv')
        if not os.path.exists(trend_path):
            run_results.append({'variable': var, 'status': 'missing trend file', 'detail': trend_path})
            continue
        df_trend = monthly_trend_inputs[var].copy()
        required_trend_columns = {'ID', 'MONTH', 'SEN_SLOPE', 'P_VALUE'}
        missing_trend_columns = required_trend_columns - set(df_trend.columns)
        if missing_trend_columns:
            run_results.append({'variable': var, 'status': 'missing columns', 'detail': str(sorted(missing_trend_columns))})
            continue
        sen_slope_decade = (pd.to_numeric(df_trend['SEN_SLOPE'], errors='coerce') * 10).dropna()
        if sen_slope_decade.empty:
            run_results.append({'variable': var, 'status': 'no valid SEN_SLOPE values', 'detail': ''})
            continue
        sen_boxplot_stats = boxplot_stats(sen_slope_decade, whis=1.5)[0]
        var_min = float(sen_boxplot_stats['whislo'])
        var_max = float(sen_boxplot_stats['whishi'])
        df_trend_update = df_trend.merge(df_info, on='ID', how='left', validate='many_to_one')
        selected_cmap = 'RdYlGn' if var.upper() == 'PRCP' else 'RdYlGn_r'
        try:
            plot_monthly_climate_rasters_with_points_updated(base_path=raster_dir, variable=var, frr_shp_path=FRR_shape_path, state_shp_path=state_shp_path, var_min=var_min, var_max=var_max, lower_tail_percentile=2.5, upper_tail_percentile=96.5 if var == 'PRCP' else 97.5, middle_classes=5, scale_factor=10, force_zero_break=True, cmap=selected_cmap, unit=trend_unit_labels.get(var, '$\\mathrm{decade}^{-1}$'), point_df=df_trend_update, point_filter_col='P_VALUE', point_filter_thresh=0.05, colorbar_spacing='uniform', title=None, save_dir=figure_dir, save_figure=True, dpi=FIGURE_DPI)
            run_results.append({'variable': var, 'status': 'completed', 'detail': f'Lower whisker={var_min:.4f}; upper whisker={var_max:.4f}'})
        except Exception as error:
            run_results.append({'variable': var, 'status': 'failed', 'detail': str(error)})
    run_results_df = pd.DataFrame(run_results)
    if len(run_results_df) != len(variables) or not run_results_df['status'].eq('completed').all():
        raise RuntimeError('Some monthly trend maps failed. Review the run summary above.')
    sos_trend_updated = dl.load_trend_data('SOS').copy()[['ID', 'TREND', 'P_VALUE', 'SEN_SLOPE']].rename(columns={'TREND': 'SOST', 'P_VALUE': 'SOSP', 'SEN_SLOPE': 'SOSS'})
    eos_trend_updated = dl.load_trend_data('EOS').copy()[['ID', 'TREND', 'P_VALUE', 'SEN_SLOPE']].rename(columns={'TREND': 'EOST', 'P_VALUE': 'EOSP', 'SEN_SLOPE': 'EOSS'})
    gsl_trend_updated = dl.load_trend_data('GSL').copy()[['ID', 'TREND', 'P_VALUE', 'SEN_SLOPE']].rename(columns={'TREND': 'GSLT', 'P_VALUE': 'GSLP', 'SEN_SLOPE': 'GSLS'})
    growing_season_trend = reduce(lambda left, right: left.merge(right, on='ID', how='outer', validate='one_to_one'), [gsl_trend_updated, sos_trend_updated, eos_trend_updated])
    growing_season_trend_update = growing_season_trend.merge(station_locations[['ID', 'Lat', 'Lon']], on='ID', how='inner', validate='one_to_one')
    import os
    import string
    import numpy as np
    import pandas as pd
    import rasterio
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from rasterio.mask import mask
    from rasterio.plot import show

    def plot_growing_season_rasters_with_points_updated(base_path, variables, frr_shp_path=None, state_shp_path=None, cmap='RdYlGn', title=None, unit_dict=None, percentile_min=2.5, percentile_max=97.5, middle_classes=5, scale_factor=10, force_zero_break=True, var_min_dict=None, var_max_dict=None, colorbar_spacing='uniform', save_folder=None, save_figure=True, dpi=FIGURE_DPI, point_df=None, point_filter_dict=None):
        """
    Plot growing-season Sen's slope rasters using exactly
    seven segmented color classes.

    DEFAULT CLASS STRUCTURE
    -----------------------

        Class 1:
            0 to 2.5 percentile

        Classes 2 to 6:
            Five equally spaced classes between the
            2.5 and 97.5 percentiles

        Class 7:
            97.5 to 100 percentile

    Therefore:

        8 boundaries
        7 colored segments

    ZERO HANDLING
    -------------

    If the trend range crosses zero and zero is not already a
    class boundary, the existing INTERNAL boundary closest to
    zero is moved to exactly zero.

    Example:

        Original boundaries:
            -5, 1, 2

        Updated boundaries:
            -5, 0, 2

    Zero is NOT added as a new boundary, so the total number
    of segments remains seven.

    COLORBAR
    --------

    Tick marks are placed directly at the class boundaries.

    For example:

        -5       0       2       5
         |-------|-------|-------|

    rather than:

        -5 to 0    0 to 2    2 to 5
    """
        if not 0 < percentile_min < percentile_max < 100:
            raise ValueError('Percentiles must satisfy: 0 < percentile_min < percentile_max < 100.')
        if middle_classes != 5:
            raise ValueError('This updated function is designed to always produce 7 segments. Therefore, middle_classes must equal 5.')
        if scale_factor <= 0:
            raise ValueError('scale_factor must be greater than zero.')
        if not variables:
            raise ValueError('At least one variable must be provided.')
        var_min_dict = {} if var_min_dict is None else var_min_dict
        var_max_dict = {} if var_max_dict is None else var_max_dict
        unit_dict = {} if unit_dict is None else unit_dict
        point_filter_dict = {} if point_filter_dict is None else point_filter_dict
        tif_name_map = {'SOSS': 'SOS_trend_SEN_SLOPE_IDW.tif', 'EOSS': 'EOS_trend_SEN_SLOPE_IDW.tif', 'GSLS': 'GSL_trend_SEN_SLOPE_IDW.tif'}
        display_name_map = {'SOSS': 'SOS', 'EOSS': 'EOS', 'GSLS': 'GSL'}
        tif_files = [os.path.join(base_path, tif_name_map.get(var, f'trend_{var}_clipped.tif')) for var in variables]
        FRR_shape = gpd.read_file(frr_shp_path)
        state_shape = gpd.read_file(state_shp_path)
        if point_df is not None:
            required_location_columns = {'Lat', 'Lon'}
            missing_location_columns = required_location_columns - set(point_df.columns)
            if missing_location_columns:
                raise ValueError(f'point_df is missing required columns: {sorted(missing_location_columns)}')
            point_df = point_df.copy()
            point_df['Lat'] = pd.to_numeric(point_df['Lat'], errors='coerce')
            point_df['Lon'] = pd.to_numeric(point_df['Lon'], errors='coerce')
        fig, axes = plt.subplots(1, len(tif_files), figsize=(7 * len(tif_files), 8))
        if len(tif_files) == 1:
            axes = [axes]
        for i, (tif_path, ax) in enumerate(zip(tif_files, axes)):
            var = variables[i]
            variable_prefix = var.upper()
            if not os.path.exists(tif_path):
                ax.set_title(f'({string.ascii_lowercase[i]}) Missing', fontsize=22, loc='left', fontweight='bold')
                ax.axis('off')
                continue
            with rasterio.open(tif_path) as src:
                FRR_shape_proj = FRR_shape.to_crs(src.crs)
                state_shape_proj = state_shape.to_crs(src.crs)
                clipped_raster, clipped_transform = mask(src, FRR_shape_proj.geometry, crop=True, filled=False)
                data = clipped_raster[0].astype(float)
                data = np.ma.masked_invalid(data)
                data = np.ma.masked_where(data < -100, data)
                data = data * scale_factor
                flat_data_original = data.compressed()
                flat_data_original = flat_data_original[np.isfinite(flat_data_original)]
                if flat_data_original.size == 0:
                    ax.axis('off')
                    continue
                actual_min = float(np.nanmin(flat_data_original))
                actual_max = float(np.nanmax(flat_data_original))
                selected_min = var_min_dict.get(var, None)
                selected_max = var_max_dict.get(var, None)
                applied_min = actual_min if selected_min is None else float(selected_min)
                applied_max = actual_max if selected_max is None else float(selected_max)
                if applied_min >= applied_max:
                    raise ValueError(f'For {var}, the applied minimum must be smaller than the applied maximum.')
                values_below_min = int(np.sum(flat_data_original < applied_min))
                values_above_max = int(np.sum(flat_data_original > applied_max))
                flat_data = np.clip(flat_data_original, applied_min, applied_max)
                plot_data = np.ma.clip(data, applied_min, applied_max)
                middle_percentile_edges = np.linspace(percentile_min, percentile_max, middle_classes + 1)
                percentile_edges = np.concatenate([[0], middle_percentile_edges, [100]])
                boundaries = np.nanpercentile(flat_data, percentile_edges)
                boundaries = np.asarray(boundaries, dtype=float)
                boundaries[0] = applied_min
                boundaries[-1] = applied_max
                expected_boundary_count = 8
                if len(boundaries) != expected_boundary_count:
                    raise RuntimeError(f'{var}: expected 8 boundaries but obtained {len(boundaries)}.')
                if np.any(np.diff(boundaries) <= 0):
                    total_range = applied_max - applied_min
                    epsilon = max(total_range * 1e-08, 1e-12)
                    for boundary_index in range(1, len(boundaries) - 1):
                        if boundaries[boundary_index] <= boundaries[boundary_index - 1]:
                            boundaries[boundary_index] = boundaries[boundary_index - 1] + epsilon
                    for boundary_index in range(len(boundaries) - 2, 0, -1):
                        if boundaries[boundary_index] >= boundaries[boundary_index + 1]:
                            boundaries[boundary_index] = boundaries[boundary_index + 1] - epsilon
                    boundaries[0] = applied_min
                    boundaries[-1] = applied_max
                if force_zero_break and applied_min < 0 and (applied_max > 0):
                    data_range = applied_max - applied_min
                    zero_tolerance = max(1e-12, data_range * 1e-09)
                    near_zero_mask = np.isclose(boundaries, 0.0, atol=zero_tolerance, rtol=0)
                    if np.any(near_zero_mask):
                        closest_zero_index = int(np.argmin(np.abs(boundaries)))
                        old_value = float(boundaries[closest_zero_index])
                        boundaries[closest_zero_index] = 0.0
                    else:
                        internal_boundaries = boundaries[1:-1]
                        closest_internal_index = int(np.argmin(np.abs(internal_boundaries)))
                        boundary_index = closest_internal_index + 1
                        old_value = float(boundaries[boundary_index])
                        boundaries[boundary_index] = 0.0
                if np.any(np.diff(boundaries) <= 0):
                    raise ValueError(f'Boundary adjustment resulted in non-increasing boundaries for {var}.')
                number_of_segments = len(boundaries) - 1
                if number_of_segments != 7:
                    raise RuntimeError(f'{var} produced {number_of_segments} segments instead of 7.')
                for class_number, (value_low, value_high) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
                    if value_high <= 0:
                        trend_direction = 'negative'
                    elif value_low >= 0:
                        trend_direction = 'positive'
                    else:
                        trend_direction = 'mixed'
                discrete_cmap = plt.get_cmap(cmap, number_of_segments)
                norm = colors.BoundaryNorm(boundaries=boundaries, ncolors=discrete_cmap.N, clip=True)

                def format_boundary_value(value):
                    if np.isclose(value, 0):
                        value = 0.0
                    return f'{value:.1f}'
                boundary_labels = [format_boundary_value(value) for value in boundaries]
                show(plot_data, transform=clipped_transform, ax=ax, cmap=discrete_cmap, norm=norm, zorder=1)
                state_shape_proj.boundary.plot(ax=ax, edgecolor='black', linewidth=1, zorder=2)
                FRR_shape_proj.boundary.plot(ax=ax, edgecolor='black', linewidth=2.5, facecolor='none', zorder=3)
                if point_df is not None and var in point_filter_dict:
                    filter_col, threshold = point_filter_dict[var]
                    if filter_col not in point_df.columns:
                        raise ValueError(f"Point DataFrame does not contain the column '{filter_col}' required for {var}.")
                    point_df[filter_col] = pd.to_numeric(point_df[filter_col], errors='coerce')
                    df_filtered = point_df.loc[point_df[filter_col] < threshold].copy()
                    df_filtered = df_filtered.dropna(subset=['Lat', 'Lon', filter_col])
                    if not df_filtered.empty:
                        gdf_points = gpd.GeoDataFrame(df_filtered, geometry=gpd.points_from_xy(df_filtered['Lon'], df_filtered['Lat']), crs='EPSG:4326').to_crs(src.crs)
                        ax.scatter(gdf_points.geometry.x, gdf_points.geometry.y, facecolors='none', edgecolors='black', linewidths=1, s=50, marker='*', alpha=0.6, zorder=4)
                ax.set_title(f'({string.ascii_lowercase[i]})', fontsize=24, loc='left', fontweight='bold')
                ax.text(0.5, 1.02, display_name_map.get(var, var), fontsize=24, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)
                ax.axis('off')
                scalar_mappable = cm.ScalarMappable(norm=norm, cmap=discrete_cmap)
                scalar_mappable.set_array([])
                cbar = plt.colorbar(scalar_mappable, ax=ax, orientation='horizontal', pad=0.05, aspect=40, boundaries=boundaries, ticks=boundaries, spacing=colorbar_spacing, drawedges=True)
                cbar.ax.set_xticklabels(boundary_labels, rotation=0, ha='center', fontsize=20, fontweight='bold')
                cbar.ax.tick_params(axis='x', which='major', length=6, width=1.3, direction='out')
                unit_label = unit_dict.get(var, '$\\mathrm{Days}\\,\\mathrm{decade}^{-1}$')
                cbar.set_label(unit_label, fontsize=22, fontweight='bold', labelpad=12)
                cbar.outline.set_linewidth(1.5)
                if cbar.dividers is not None:
                    cbar.dividers.set_linewidth(1.2)
        if title:
            fig.suptitle(title, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        if save_folder and save_figure:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, 'growing_season_trend_segmented_with_points_updated.jpg')
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    base_path = dl.RASTER_DIR
    FRR_shape_path = dl.FRR_SHAPE_PATH
    state_shp_path = dl.CONUS_SHAPE_PATH
    save_folder = MAP_OUTPUT_DIR
    var_min_dict = {'SOSS': -5, 'EOSS': -5, 'GSLS': -10}
    var_max_dict = {'SOSS': 3, 'EOSS': 11, 'GSLS': 14}
    unit_dict = {'SOSS': '$\\mathrm{Days}\\,\\mathrm{decade}^{-1}$', 'EOSS': '$\\mathrm{Days}\\,\\mathrm{decade}^{-1}$', 'GSLS': '$\\mathrm{Days}\\,\\mathrm{decade}^{-1}$'}
    point_filter_dict = {'SOSS': ('SOSP', 0.05), 'EOSS': ('EOSP', 0.05), 'GSLS': ('GSLP', 0.05)}
    plot_growing_season_rasters_with_points_updated(base_path=base_path, variables=['SOSS', 'EOSS', 'GSLS'], frr_shp_path=FRR_shape_path, state_shp_path=state_shp_path, cmap='RdYlGn', unit_dict=unit_dict, percentile_min=2.5, percentile_max=97.5, middle_classes=5, scale_factor=10, force_zero_break=True, var_min_dict=var_min_dict, var_max_dict=var_max_dict, colorbar_spacing='uniform', title=None, save_folder=save_folder, save_figure=True, dpi=FIGURE_DPI, point_df=growing_season_trend_update, point_filter_dict=point_filter_dict)
    expected_figures = [MAP_OUTPUT_DIR / f'{var.upper()}_monthly_avg_segmented_updated.jpg' for var in variables]
    expected_figures += [MAP_OUTPUT_DIR / f'{var.upper()}_monthly_trend_segmented_updated.jpg' for var in variables]
    expected_figures.append(MAP_OUTPUT_DIR / 'growing_season_trend_segmented_with_points_updated.jpg')
    missing_outputs = [path.name for path in expected_figures if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f'Missing figure outputs: {missing_outputs}')
    print(f'Saved all {len(expected_figures)} map figures to {MAP_OUTPUT_DIR}')


if __name__ == "__main__":
    # Unexpected exceptions and the original failed-map checks still propagate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
