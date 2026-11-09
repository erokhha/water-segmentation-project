from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from scipy import ndimage


def derive_valid_footprint(
    image_path: Path,
    fill_rgb: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    if not isinstance(fill_rgb, tuple) or len(fill_rgb) != 3 or any(
        not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255
        for value in fill_rgb
    ):
        raise ValueError('fill_rgb must contain three uint8 channel values')
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with rasterio.open(image_path) as source:
        if source.count != 3 or source.dtypes != ('uint8',) * 3 or \
                source.colorinterp != (ColorInterp.red, ColorInterp.green, ColorInterp.blue) or \
                source.width <= 0 or source.height <= 0:
            raise ValueError('Expected three uint8 RGB bands with positive dimensions')
        fill = np.empty((source.height, source.width), dtype=bool)
        color = np.asarray(fill_rgb, dtype=np.uint8)[:, None, None]
        for y0 in range(0, source.height, 256):
            height = min(256, source.height - y0)
            rgb = source.read([1, 2, 3], window=Window(0, y0, source.width, height))
            fill[y0:y0 + height] = np.all(rgb == color, axis=0)

    border = np.zeros_like(fill)
    border[0] = fill[0]
    border[-1] = fill[-1]
    border[:, 0] = fill[:, 0]
    border[:, -1] = fill[:, -1]
    outside = ndimage.binary_propagation(
        border, structure=ndimage.generate_binary_structure(2, 1), mask=fill)
    valid = ~outside
    if not valid.any():
        raise ValueError('The image contains no valid pixels outside the exterior fill')
    return valid


def build_footprint_filtered_result(
    source_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError('Output directory must be absent or empty')
    source_manifest_path = Path(source_manifest_path).expanduser().resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    if source_manifest.get('status') != 'complete' or \
            source_manifest.get('purpose') != 'diagnostic_blended' or \
            source_manifest.get('tile_size') != 512 or \
            source_manifest.get('stride') != 384 or \
            source_manifest.get('threshold') not in (.65, .35):
        raise ValueError('Source manifest is not the fixed blended diagnostic result')
    if not all(isinstance(source_manifest.get(key), int) and source_manifest[key] > 0
               for key in ('width', 'height')):
        raise ValueError('Source manifest dimensions are invalid')
    threshold = source_manifest['threshold']
    image_path = Path(source_manifest['image']).expanduser().resolve()
    probability_path = Path(source_manifest['probability']).expanduser().resolve()
    old_mask_path = Path(source_manifest['mask']).expanduser().resolve()
    for path in (image_path, probability_path, old_mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    image_hash = file_hash(image_path)
    if image_hash != source_manifest.get('image_sha256'):
        raise ValueError('Input RGB SHA-256 differs from the diagnostic manifest')
    probability_hash = file_hash(probability_path)
    old_mask_hash = file_hash(old_mask_path)
    width, height = source_manifest['width'], source_manifest['height']
    with rasterio.open(image_path) as image, rasterio.open(probability_path) as probability, \
            rasterio.open(old_mask_path) as old_mask:
        if (image.width, image.height, image.count, image.dtypes, image.crs) != \
                (width, height, 3, ('uint8',) * 3, None) or \
                image.colorinterp != (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
            raise ValueError('Source RGB metadata differs from manifest')
        for raster, dtype in ((probability, 'float32'), (old_mask, 'uint8')):
            if (raster.width, raster.height, raster.count, raster.dtypes, raster.crs) != \
                    (width, height, 1, (dtype,), None):
                raise ValueError('Diagnostic raster metadata differs from RGB')
        water_before = 0
        for y0 in range(0, height, 256):
            window = Window(0, y0, width, min(256, height - y0))
            values = probability.read(1, window=window)
            binary = old_mask.read(1, window=window)
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)) or \
                    not np.array_equal(binary, (values >= threshold).astype(np.uint8)):
                raise ValueError('Saved mask differs from the thresholded probability')
            water_before += int(binary.sum())

    valid = derive_valid_footprint(image_path)
    outside_pixels = int(np.count_nonzero(~valid))
    internal_fill_preserved = 0
    with rasterio.open(image_path) as image:
        for y0 in range(0, height, 256):
            h = min(256, height - y0)
            rgb = image.read([1, 2, 3], window=Window(0, y0, width, h))
            internal_fill_preserved += int(np.count_nonzero(
                valid[y0:y0 + h] & np.all(rgb == 1, axis=0)))

    output_dir.mkdir(parents=True, exist_ok=True)
    valid_path = output_dir / 'valid_footprint.tif'
    mask_path = output_dir / 'water_mask.tif'
    profile = dict(driver='GTiff', width=width, height=height, count=1,
                   dtype='uint8', compress='deflate')
    if width >= 512 and height >= 512:
        profile.update(tiled=True, blockxsize=512, blockysize=512)
    water_after = 0
    with rasterio.open(old_mask_path) as old_mask, \
            rasterio.open(valid_path, 'w', **profile) as valid_file, \
            rasterio.open(mask_path, 'w', nodata=255, **profile) as mask_file:
        for y0 in range(0, height, 256):
            h = min(256, height - y0)
            window = Window(0, y0, width, h)
            local_valid = valid[y0:y0 + h]
            binary = old_mask.read(1, window=window)
            result = np.where(local_valid, binary, 255).astype(np.uint8)
            valid_file.write(local_valid.astype(np.uint8), 1, window=window)
            mask_file.write(result, 1, window=window)
            water_after += int(np.count_nonzero(result == 1))

    with rasterio.open(valid_path) as valid_file, rasterio.open(mask_path) as mask_file, \
            rasterio.open(old_mask_path) as old_mask:
        if (valid_file.width, valid_file.height, valid_file.dtypes, valid_file.crs) != \
                (width, height, ('uint8',), None) or \
                (mask_file.width, mask_file.height, mask_file.dtypes, mask_file.crs,
                 mask_file.nodata) != (width, height, ('uint8',), None, 255):
            raise RuntimeError('Saved footprint or water mask metadata is invalid')
        counted_outside = 0
        counted_water = 0
        for y0 in range(0, height, 256):
            h = min(256, height - y0)
            window = Window(0, y0, width, h)
            local_valid = valid[y0:y0 + h]
            saved_valid = valid_file.read(1, window=window)
            saved_water = mask_file.read(1, window=window)
            original = old_mask.read(1, window=window)
            if not np.array_equal(saved_valid, local_valid.astype(np.uint8)) or \
                    not np.array_equal(saved_water,
                                       np.where(local_valid, original, 255).astype(np.uint8)):
                raise RuntimeError('Saved footprint or water mask differs from inputs')
            counted_outside += int(np.count_nonzero(saved_valid == 0))
            counted_water += int(np.count_nonzero(saved_water == 1))
        if counted_outside != outside_pixels or counted_water != water_after:
            raise RuntimeError('Saved raster counts differ from manifest counts')
    if file_hash(image_path) != image_hash or file_hash(probability_path) != probability_hash or \
            file_hash(old_mask_path) != old_mask_hash:
        raise RuntimeError('Input raster changed during footprint processing')

    result = {
        'status': 'complete', 'purpose': 'footprint_filtered_diagnostic',
        'source_manifest': str(source_manifest_path),
        'source_manifest_sha256': file_hash(source_manifest_path),
        'image': str(image_path), 'image_sha256': image_hash,
        'checkpoint_sha256': source_manifest.get('checkpoint_sha256'),
        'probability': str(probability_path), 'probability_sha256': probability_hash,
        'source_mask': str(old_mask_path), 'source_mask_sha256': old_mask_hash,
        'valid_footprint': str(valid_path), 'valid_footprint_sha256': file_hash(valid_path),
        'water_mask': str(mask_path), 'water_mask_sha256': file_hash(mask_path),
        'code_sha256': file_hash(Path(__file__)),
        'width': width, 'height': height, 'threshold': threshold,
        'fill_rgb': [1, 1, 1], 'connectivity': 4,
        'valid_pixels': int(valid.size - outside_pixels), 'outside_pixels': outside_pixels,
        'internal_fill_pixels_preserved': internal_fill_preserved,
        'water_pixels_before': water_before, 'water_pixels_after': water_after,
        'removed_positive_pixels': water_before - water_after,
        'not_final_vector_layer': True,
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_footprint_filtered_result(args.source_manifest, args.output),
                     ensure_ascii=False, indent=2))
