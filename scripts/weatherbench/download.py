#!/usr/bin/env python3
"""
Simple WeatherBench downloader: minimal, optimized for fast local disk access.
Downloads variables and time slices from GCP Zarr to local Zarr with proper coordinates.
"""

import argparse
import logging
from pathlib import Path

import xarray as xr
import zarr
from tqdm.dask import TqdmCallback

import zephyr
from zephyr.data.variables import (
    ATMOSPHERIC_VARIABLE_NAMES,
    ATOMOSPHERIC_LEVELS,
    FORCED_VARIABLES,
    SURFACE_VARIABLES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCP_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
DEFAULT_OUTPUT_PATH = Path(zephyr.__file__).parents[1] / "data" / "weatherbench.zarr"


def open_weatherbench_dataset() -> xr.Dataset:
    """Open the full WeatherBench dataset from GCP."""
    return xr.open_zarr(GCP_PATH, chunks="auto", consolidated=True)


def download_weatherbench(
    dataset: xr.Dataset,
    variables: list[str],
    years: list[int],
    pressure_levels: list[int],
    output_path: Path,
):
    """Simple download with level selection, optimized chunking and compression."""

    logger.info(f"Available variables: {list(dataset.data_vars.keys())}")
    logger.info(f"Available coordinates: {list(dataset.coords.keys())}")

    start_year, end_year = years
    logger.info(f"Filtering variables: {variables}")
    logger.info(f"Filtering years: {start_year}-{end_year}")

    ds_filtered = dataset[variables].sel(time=slice(str(start_year), str(end_year)))

    # Filter pressure levels if specified
    if pressure_levels and "level" in ds_filtered.coords:
        logger.info(f"Filtering pressure levels: {pressure_levels}")
        ds_filtered = ds_filtered.sel(level=pressure_levels)
    elif pressure_levels:
        logger.info("No pressure levels to filter (surface variables only)")

    logger.info("Final dataset shape:")
    for var in ds_filtered.data_vars:
        logger.info(f"  {var}: {ds_filtered[var].shape} {ds_filtered[var].dims}")

    chunk_config = {
        "time": 8,
        "level": -1,
        "latitude": -1,
        "longitude": -1,
    }
    ds_filtered = ds_filtered.chunk(chunk_config)

    # Compression encoding for fast disk reads
    encoding = {
        var: {"compressor": zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.SHUFFLE)}
        for var in ds_filtered.data_vars
    }

    logger.info(f"Saving to {output_path} with zstd compression")
    with TqdmCallback(desc="Downloading weatherbench"):
        ds_filtered.to_zarr(output_path, encoding=encoding)

    logger.info(f"Successfully downloaded {len(variables)} variables to {output_path}")
    logger.info(f"Coordinates preserved: {list(ds_filtered.coords.keys())}")


def parse_args():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Simple WeatherBench downloader with optimal chunking"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Local output path",
    )
    parser.add_argument("--years", default="2000,2020", help="Year range as start,end")

    return parser.parse_args()


def main():
    """Entry point for simple download script."""
    args = parse_args()
    year_range = list(map(int, args.years.split(",")))

    if len(year_range) != 2:
        raise ValueError("Years must be specified as start,end (e.g., 2010,2014)")

    years = year_range
    try:
        dataset = open_weatherbench_dataset()
        download_weatherbench(
            dataset=dataset,
            variables=SURFACE_VARIABLES + ATMOSPHERIC_VARIABLE_NAMES + FORCED_VARIABLES,
            years=years,
            pressure_levels=ATOMOSPHERIC_LEVELS,
            output_path=Path(args.output),
        )
        logger.info("Download completed successfully!")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    main()
