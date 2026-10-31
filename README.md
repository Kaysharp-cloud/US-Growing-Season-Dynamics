
# Seasonally Asymmetric Growing-Season Lengthening and Contrasting Thermal–Precipitation Trends Across U.S. Agricultural Regions, 1996–2025

Python analysis scripts supporting the manuscript **"Seasonally Asymmetric Growing-Season Lengthening and Contrasting Thermal–Precipitation Trends Across U.S. Agricultural Regions, 1996–2025."**

## 📁 Repository Structure

The following directory structure contains the Python analysis scripts and the data directories expected during execution.

```text
.
├── README.md
├── scripts/
│   ├── data_loader.py
│   ├── growing_season.py
│   ├── monthly_climate_indices.py
│   ├── regional_growing_season_trends.py
│   ├── supplementary_trend_tables.py
│   ├── arcpy_raster_generation.py
│   ├── climate_spatial_maps.py
│   └── python_script_sources.txt
└── data/
    └── raw_data/
        ├── HPC_data/
        ├── FRR_shapefile/
        └── US_CONUS_Shapefile/
```

## 🐍 Python Environment

To execute the analysis scripts and visualize existing raster datasets, install the required Python packages:

```bash
python -m pip install numpy pandas pymannkendall matplotlib rasterio geopandas
```


The `arcpy_raster_generation.py` script requires **ArcGIS Pro**, including access to **ArcPy** and the **Spatial Analyst extension**.

### Option 1: Run Directly in ArcGIS Pro (Recommended)

1. Open **ArcGIS Pro** and ensure the Spatial Analyst extension is available.
2. Navigate to **View → Python Window**.
3. Execute the script using:

```python
exec(open(r"C:\path\to\repository\scripts\arcpy_raster_generation.py", encoding="utf-8").read())
```


### Option 2: Run Using the ArcGIS Pro Python Environment

The script can also be executed using the Python interpreter supplied with ArcGIS Pro, provided that ArcPy and the Spatial Analyst license are available.

Open the **ArcGIS Pro Python Command Prompt**, navigate to the script directory, and run:

```bash
python arcpy_raster_generation.py
```



