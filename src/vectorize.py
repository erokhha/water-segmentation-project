"""Exact water-mask polygonization in local pixel coordinates."""
from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape


LOCAL_TRANSFORM = Affine(1, 0, 0, 0, -1, 0)
_BATCH_SIZE = 128


def vectorize_water_mask(
    mask_path: Path,
    output_path: Path,
    *,
    layer: str = "water",
) -> dict[str, int]:
    """Write exact four-connected water polygons from a 0/1/255 TIFF."""
    mask_path = Path(mask_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if output_path.suffix.lower() != ".gpkg" or not isinstance(layer, str) or not layer.strip():
        raise ValueError("Expected a .gpkg output and a nonempty layer name")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(output_path.parent)

    with rasterio.open(mask_path) as source:
        if (
            source.driver != "GTiff"
            or source.count != 1
            or source.dtypes != ("uint8",)
            or source.nodata != 255
            or source.crs is not None
            or source.transform != Affine.identity()
            or source.width <= 0
            or source.height <= 0
        ):
            raise ValueError("Expected an unreferenced, single-band uint8 TIFF with NoData=255")
        pixels = source.read(1)

    counts = np.bincount(pixels.reshape(-1), minlength=256)
    if np.any(counts[2:255]):
        raise ValueError("Mask may contain only values 0, 1 and 255")
    expected_water = int(counts[1])

    polygon_count = 0
    water_pixels = 0
    geometries = []
    areas = []
    output_started = False
    try:
        for geometry_json, value in shapes(
            pixels,
            mask=pixels == 1,
            connectivity=4,
            transform=LOCAL_TRANSFORM,
        ):
            if value != 1:
                raise RuntimeError("Polygonization returned a non-water value")
            geometry = shape(geometry_json)
            area = geometry.area
            pixel_area = int(area)
            if (
                geometry.geom_type != "Polygon"
                or not geometry.is_valid
                or area <= 0
                or area != pixel_area
            ):
                raise RuntimeError("Polygonization produced an invalid pixel polygon")
            geometries.append(geometry)
            areas.append(pixel_area)
            polygon_count += 1
            water_pixels += pixel_area
            if len(geometries) == _BATCH_SIZE:
                frame = gpd.GeoDataFrame(
                    {"class_name": ["water"] * len(geometries),
                     "pixel_area": np.asarray(areas, dtype=np.int64)},
                    geometry=geometries,
                    crs=None,
                )
                output_started = True
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="'crs' was not provided")
                    pyogrio.write_dataframe(
                        frame, output_path, layer=layer, driver="GPKG",
                        geometry_type="Polygon", append=output_path.exists(),
                    )
                geometries.clear()
                areas.clear()

        if water_pixels != expected_water:
            raise RuntimeError("Polygon area differs from the number of water pixels")
        if geometries or not output_path.exists():
            frame = gpd.GeoDataFrame(
                {"class_name": ["water"] * len(geometries),
                 "pixel_area": np.asarray(areas, dtype=np.int64)},
                geometry=geometries,
                crs=None,
            )
            output_started = True
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="'crs' was not provided")
                pyogrio.write_dataframe(
                    frame, output_path, layer=layer, driver="GPKG",
                    geometry_type="Polygon", append=output_path.exists(),
                )

        info = pyogrio.read_info(output_path, layer=layer)
        if (
            info["features"] != polygon_count
            or info["crs"] is not None
            or info["geometry_type"] != "Polygon"
            or set(info["fields"]) != {"class_name", "pixel_area"}
        ):
            raise RuntimeError("Saved GeoPackage metadata differs from the water mask contract")
    except Exception:
        if output_started:
            output_path.unlink(missing_ok=True)
        raise

    return {"polygon_count": polygon_count, "water_pixels": water_pixels}
