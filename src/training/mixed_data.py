"""Verified 512 / 768 training data readers, consolidated from legacy code.

Existing per-source algorithms and manifest checks are retained. The old
training-data modules are not required when importing this module.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.windows import Window
from torch.utils.data import Dataset

from dataset import MEAN, STD, WATER_CLASS, WaterDataset
from geometry_augmentation import augment_triplet
from training.sources import checksum
from training.losses import negative_weighted_bce_12
from training.runner import configure_training_mode, state_hash

# Independent 512 / 768 readers

# Original: src/inspect_dataset.py :: load_pair
def _inspect_load_pair(image_path: Path, mask_path: Path):
    with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
        if image_src.count != 3 or image_src.dtypes != ('uint8',) * 3:
            raise ValueError('Ожидалось RGB из 3 каналов uint8; автоматического преобразования нет')
        if mask_src.count != 1 or mask_src.dtypes != ('uint8',):
            raise ValueError('Ожидалась одноканальная маска uint8 с номерами классов')
        if (image_src.width, image_src.height) != (mask_src.width, mask_src.height):
            raise ValueError('Размеры изображения и маски не совпадают')
        if image_src.crs is not None and mask_src.crs is not None:
            if image_src.crs != mask_src.crs or not image_src.transform.almost_equals(mask_src.transform):
                raise ValueError('Геопривязка изображения и маски различается')
        rgb = image_src.read()
        labels = mask_src.read(1)
        codes = np.unique(labels)
        if np.any(codes > 8):
            raise ValueError(f'Неожиданные коды разметки: {codes.tolist()}; ожидались 0..8')
        valid = (labels != 0) & (mask_src.read_masks(1) > 0)
        valid &= np.all(image_src.read_masks() > 0, axis=0)
        colors = ','.join((color.name for color in image_src.colorinterp))
    return (rgb, labels, valid, colors)


# Original: src/hard_train_tyrolw.py :: OLD_FOLDER
_tyrol_OLD_FOLDER = Path('configs/trial_train_hard_20261029_001')


# Original: src/hard_train_tyrolw.py :: OLD_TRAIN_SHA
_tyrol_OLD_TRAIN_SHA = 'd86f771b006d2eb561d7ad3cf2edb429bef11221ef4d1732651059d16147c058'


# Original: src/hard_train_tyrolw.py :: OLD_OEM_SHA
_tyrol_OLD_OEM_SHA = '8c89faaaf76bf0641593576036fb9c86e2ce1177e736fa1d6484f38a76721f2c'


# Original: src/hard_train_tyrolw.py :: OLD_VAL_SHA
_tyrol_OLD_VAL_SHA = '91a8dac45211a4cd43fc14175619492046b312bfd76c5f896b98f1f7b3714709'


# Original: src/hard_train_tyrolw.py :: OLD_TEST_SHA
_tyrol_OLD_TEST_SHA = '60c77b545d69b11b582c35711ef55284937e70824bb91eb859eacbb786f62906'


# Original: src/hard_train_tyrolw.py :: APPROVED_EXCLUSIONS
_tyrol_APPROVED_EXCLUSIONS = frozenset({('OEM', 'monrovia/monrovia_44'), ('OEM', 'monrovia/monrovia_47'), ('OEM', 'monrovia/monrovia_54'), ('FLAIR', 'D013_2020:aerial/Z10_UU/IMG_009608.tif'), ('FLAIR', 'D013_2020:aerial/Z10_UU/IMG_009618.tif'), ('FLAIR', 'D013_2020:aerial/Z10_UU/IMG_009629.tif'), ('FLAIR', 'D013_2020:aerial/Z10_UU/IMG_009639.tif'), ('FLAIR', 'D013_2020:aerial/Z10_UU/IMG_009681.tif')})


# Original: src/hard_train_tyrolw.py :: EXPECTED_COUNTS
_tyrol_EXPECTED_COUNTS = {'OEM': 1916, 'CAID': 1461, 'LandCover': 69, 'xBD': 54, 'FLAIR': 523}


# Original: src/hard_train_tyrolw.py :: EXPECTED_FOCUS
_tyrol_EXPECTED_FOCUS = {'OEM': 959, 'CAID': 455, 'LandCover': 1, 'xBD': 54, 'FLAIR': 523}


# Original: src/hard_train_tyrolw.py :: _read_csv
def _tyrol__read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


# Original: src/hard_train_tyrolw.py :: partition_tyrolw
def _tyrol_partition_tyrolw(rows: list[dict], footprints: dict[str, tuple[float, float, float, float]]) -> tuple[set[str], set[str]]:
    """Keep every touching component containing an author-Val tile in Validation."""
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(footprints) or (not ids) or any((r['source_split'] not in {'train', 'val'} for r in rows)) or (not any((r['source_split'] == 'val' for r in rows))):
        raise ValueError('Invalid tyrolw split inputs')
    for bounds in footprints.values():
        if len(bounds) != 4 or not all((math.isfinite(x) for x in bounds)) or bounds[2] <= bounds[0] or (bounds[3] <= bounds[1]):
            raise ValueError('Invalid tyrolw footprint')
    parent = list(range(len(ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for i, left_id in enumerate(ids):
        a = footprints[left_id]
        for j in range(i):
            b = footprints[ids[j]]
            x_gap = max(0.0, a[0] - b[2], b[0] - a[2])
            y_gap = max(0.0, a[1] - b[3], b[1] - a[3])
            if x_gap <= 1e-05 and y_gap <= 1e-05:
                parent[find(i)] = find(j)
    held_roots = {find(i) for i, r in enumerate(rows) if r['source_split'] == 'val'}
    val = {sample_id for i, sample_id in enumerate(ids) if find(i) in held_roots}
    return (set(ids) - val, val)


# Original: src/hard_train_tyrolw.py :: _tyrolw_footprints
def _tyrol__tyrolw_footprints(project: Path, rows: list[dict]) -> dict[str, tuple]:
    root = project / 'data/raw/openearthmap'
    footprints = {}
    for row in rows:
        if not row['mask']:
            raise ValueError('Missing tyrolw mask')
        image_path, mask_path = (root / row['image'], root / row['mask'])
        with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
            if image.crs is None or image.crs.to_epsg() != 31254 or mask.crs != image.crs or (not mask.transform.almost_equals(image.transform)) or (image.width != 1000) or (image.height != 1000) or (mask.width != image.width) or (mask.height != image.height) or (image.count != 3) or (mask.count != 1) or (mask.dtypes != ('uint8',)) or (abs(image.transform.a - 0.3) > 1e-06) or (abs(image.transform.e + 0.3) > 1e-06):
                raise ValueError(f"Tyrolw image/mask grid differs: {row['sample_id']}")
            b = image.bounds
            footprints[row['sample_id']] = (b.left, b.bottom, b.right, b.top)
        rgb, labels, valid, _ = _inspect_load_pair(image_path, mask_path)
        rgb_digest = hashlib.sha256()
        rgb_digest.update(str(rgb.shape).encode('ascii'))
        rgb_digest.update(rgb.tobytes())
        if rgb_digest.hexdigest() != row['image_sha256'] or hashlib.sha256(labels.tobytes()).hexdigest() != row['label_sha256'] or int(valid.sum()) != int(row['valid_pixels']) or (int(((labels == 6) & valid).sum()) != int(row['water_pixels'])):
            raise ValueError(f"Tyrolw source pixels/counts differ: {row['sample_id']}")
    return footprints


# Original: src/hard_train_tyrolw.py :: _inputs
def _tyrol__inputs(project: Path):
    old = project / _tyrol_OLD_FOLDER
    paths = (old / 'train_manifest.csv', project / 'outputs/checks/oem_owner_filter_20261028_001/train_filtered.csv', project / 'configs/splits/val.csv', project / 'configs/splits/test.csv')
    if tuple((checksum(p) for p in paths)) != (_tyrol_OLD_TRAIN_SHA, _tyrol_OLD_OEM_SHA, _tyrol_OLD_VAL_SHA, _tyrol_OLD_TEST_SHA):
        raise ValueError('Frozen source Train/Validation/Test identity differs')
    train, oem, val, test = (_tyrol__read_csv(p) for p in paths)
    if (len(train), len(oem), len(val)) != (4000, 1888, 402):
        raise ValueError('Frozen source row counts differ')
    old_config = json.loads((old / 'manifest.json').read_text(encoding='utf-8'))
    return (train, oem, val, test, old_config)


# Original: src/hard_train_tyrolw.py :: _expected
def _tyrol__expected(project: Path):
    train, oem, val, test, old_config = _tyrol__inputs(project)
    original_ids = {(r['source'], r['pair_id']) for r in train}
    if len(original_ids) != 4000 or not _tyrol_APPROVED_EXCLUSIONS <= original_ids:
        raise ValueError('Approved exclusion identity differs')
    tyrolw = [r for r in val if r['region'] == 'tyrolw']
    if len(tyrolw) != 49 or Counter((r['source_split'] for r in tyrolw)) != {'train': 42, 'val': 7}:
        raise ValueError('Tyrolw source membership differs')
    footprints = _tyrol__tyrolw_footprints(project, tyrolw)
    promoted, held = _tyrol_partition_tyrolw(tyrolw, footprints)
    if (len(promoted), len(held)) != (31, 18) or any((r['sample_id'] not in held for r in tyrolw if r['source_split'] == 'val')):
        raise ValueError('Tyrolw spatial split differs')
    val_by_id = {r['sample_id']: r for r in tyrolw}
    retained = [r for r in train if (r['source'], r['pair_id']) not in _tyrol_APPROVED_EXCLUSIONS]
    added = [{'source': 'OEM', 'pair_id': sample_id, 'image': '', 'mask': '', 'source_group': 'OEM:region:tyrolw', 'focus_hard': '0', 'hard_reason': 'not_screened', 'valid_pixels': val_by_id[sample_id]['valid_pixels'], 'water_pixels': val_by_id[sample_id]['water_pixels'], 'tp_at_065': '', 'fp_at_065': '', 'fn_at_065': '', 'mask_review_status': 'technical_checked_semantic_not_owner_reviewed', 'old_route_role': 'moved_from_previous_oem_validation'} for sample_id in sorted(promoted)]
    new_train = retained + added
    new_oem = [r for r in oem if ('OEM', r['sample_id']) not in _tyrol_APPROVED_EXCLUSIONS]
    new_oem += [val_by_id[sample_id] for sample_id in sorted(promoted)]
    new_val = [r for r in val if r['sample_id'] not in promoted]
    counts = Counter((r['source'] for r in new_train))
    focus = Counter((r['source'] for r in new_train if r['focus_hard'] == '1'))
    oem_train_ids = {r['sample_id'] for r in new_oem}
    val_ids = {r['sample_id'] for r in new_val}
    test_ids = {r['sample_id'] for r in test}
    if len(new_train) != 4023 or len({(r['source'], r['pair_id']) for r in new_train}) != 4023 or dict(counts) != _tyrol_EXPECTED_COUNTS or (dict(focus) != _tyrol_EXPECTED_FOCUS) or (len(new_oem) != 1916) or (len(new_val) != 371) or (len(oem_train_ids) != 1916) or (len(val_ids) != 371) or (oem_train_ids & (val_ids | test_ids)) or (val_ids & test_ids) or (oem_train_ids != {r['pair_id'] for r in new_train if r['source'] == 'OEM'}):
        raise ValueError('New Train/Validation membership differs')
    return (new_train, new_oem, new_val, promoted, held, old_config, footprints)


# Original: src/hard_train_tyrolw.py :: load_revision
def _tyrol_load_revision(project: Path, folder: Path) -> tuple[list[dict], dict]:
    """Verify the versioned data identity before a future training entrypoint uses it."""
    project, folder = (Path(project).resolve(), Path(folder).resolve())
    if not folder.is_relative_to(project):
        raise ValueError('Manifest outside project')
    config = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    expected, oem, val, promoted, held, old, _ = _tyrol__expected(project)
    if config.get('schema_version') != 2 or config.get('train_unique_pairs') != 4023 or config.get('focus_hard_pairs') != 1992 or (config.get('source_counts') != _tyrol_EXPECTED_COUNTS) or (config.get('focus_counts') != _tyrol_EXPECTED_FOCUS) or (config.get('approved_exclusions') != [list(item) for item in sorted(_tyrol_APPROVED_EXCLUSIONS)]) or (config.get('tyrolw_promoted_train_ids') != sorted(promoted)) or (config.get('tyrolw_held_validation_ids') != sorted(held)) or (config.get('source_train_manifest_sha256') != _tyrol_OLD_TRAIN_SHA) or (config.get('source_oem_train_sha256') != _tyrol_OLD_OEM_SHA) or (config.get('source_oem_val_sha256') != _tyrol_OLD_VAL_SHA) or (config.get('source_oem_test_sha256') != _tyrol_OLD_TEST_SHA) or (config.get('oem_train') != str((folder / 'oem_train.csv').relative_to(project))) or (config.get('oem_val') != str((folder / 'oem_val.csv').relative_to(project))) or (config.get('caid_selected_index_sha256') != old['caid_selected_index_sha256']) or (config.get('flair_selected_index_sha256') != old['flair_selected_index_sha256']):
        raise ValueError('Revised manifest identity differs')
    for filename, key, content in (('train_manifest.csv', 'train_manifest_sha256', expected), ('oem_train.csv', 'oem_train_sha256', oem), ('oem_val.csv', 'oem_val_sha256', val)):
        path = folder / filename
        if checksum(path) != config.get(key) or _tyrol__read_csv(path) != content:
            raise ValueError(f'Revised content differs: {filename}')
    return ([{**row, 'focus_hard': int(row['focus_hard'])} for row in expected], config)


# Original: src/caid_training_data.py :: CAIDTrainDataset
class _caid_CAIDTrainDataset(Dataset):
    """Only vetted author-Train pairs; ZIP is read without extraction."""

    def __init__(self, archive_path: Path, index_path: Path, archive_sha256: str, index_sha256: str):
        self.archive_path = Path(archive_path).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        for path in (self.archive_path, self.index_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if checksum(self.archive_path) != archive_sha256 or checksum(self.index_path) != index_sha256:
            raise ValueError('CAID archive or eligible index SHA-256 differs')
        with self.index_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ['id', 'water_pixels', 'water_fraction']:
                raise ValueError('CAID index columns differ')
            self.rows = list(reader)
        identities = [row['id'] for row in self.rows]
        if not identities or len(identities) != len(set(identities)):
            raise ValueError('CAID index is empty or contains duplicate IDs')
        for row in self.rows:
            pixels = int(row['water_pixels'])
            fraction = float(row['water_fraction'])
            if not 0 <= pixels <= 250000 or not np.isfinite(fraction) or abs(fraction - pixels / 250000) > 1e-09:
                raise ValueError('CAID water counts are invalid')
        self.archive = zipfile.ZipFile(self.archive_path)
        try:
            split = {part: {line.strip() for line in self.archive.read(f'CAID/ImageSets/Segmentation/{part}.txt').decode('utf-8').splitlines() if line.strip()} for part in ('train', 'val', 'test')}
            if not set(identities) <= split['train'] or set(identities) & (split['val'] | split['test']):
                raise ValueError('CAID index contains an author holdout ID')
            files = set(self.archive.namelist())
            for identity in identities:
                if not identity.startswith('img_') or not identity[4:].isdigit():
                    raise ValueError('CAID ID format is invalid')
                for folder in ('JPEGImages', 'SegmentationClass'):
                    if f'CAID/{folder}/{identity}.png' not in files:
                        raise FileNotFoundError(f'CAID/{folder}/{identity}.png')
        except Exception:
            self.archive.close()
            raise

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(index, int) or index < 0 or index >= len(self.rows):
            raise IndexError(index)
        identity = self.rows[index]['id']
        rgb = Image.open(io.BytesIO(self.archive.read(f'CAID/JPEGImages/{identity}.png')))
        label = Image.open(io.BytesIO(self.archive.read(f'CAID/SegmentationClass/{identity}.png')))
        if (rgb.mode, rgb.size, rgb.format) != ('RGB', (500, 500), 'PNG') or (label.mode, label.size, label.format) != ('L', (500, 500), 'PNG'):
            raise ValueError(f'CAID pair format is invalid: {identity}')
        image_values = np.asarray(rgb, dtype=np.uint8)
        mask_values = np.asarray(label)
        if not np.all((mask_values == 0) | (mask_values == 1)):
            raise ValueError(f'CAID mask has unknown values: {identity}')
        if int(np.count_nonzero(mask_values)) != int(self.rows[index]['water_pixels']):
            raise ValueError(f'CAID mask differs from audited index: {identity}')
        image = np.zeros((3, 512, 512), dtype=np.float32)
        water = np.zeros((1, 512, 512), dtype=np.float32)
        valid = np.zeros((1, 512, 512), dtype=np.float32)
        pixels = np.transpose(image_values, (2, 0, 1)).astype(np.float32) / 255.0
        image[:, :500, :500] = (pixels - MEAN) / STD
        water[0, :500, :500] = mask_values
        valid[0, :500, :500] = 1.0
        image, water, valid = augment_triplet(image, water, valid)
        return (torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(water)), torch.from_numpy(np.ascontiguousarray(valid)))

    def __del__(self):
        archive = getattr(self, 'archive', None)
        if archive is not None:
            archive.close()


# Original: src/landcover_hard_data.py :: SIZE
_land_SIZE = 512


# Original: src/landcover_hard_data.py :: INDEX_COLUMNS
_land_INDEX_COLUMNS = ['id', 'source', 'x', 'y', 'water_pixels', 'black_pixels', 'baseline_fp']


# Original: src/landcover_hard_data.py :: _split_ids
def _land__split_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate author split ID: {path}')
    return set(values)


# Original: src/landcover_hard_data.py :: LandCoverHardDryDataset
class _land_LandCoverHardDryDataset(Dataset):

    def __init__(self, root: Path, index_path: Path, manifest_path: Path, expected_index_sha256: str, expected_manifest_sha256: str):
        self.root = Path(root).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        for path in (self.index_path, self.manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if checksum(self.index_path) != expected_index_sha256 or checksum(self.manifest_path) != expected_manifest_sha256:
            raise ValueError('Hard dry index or manifest SHA-256 differs')
        manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if manifest['hard_dry_index_sha256'] != expected_index_sha256:
            raise ValueError('Manifest points to another index')
        train = _land__split_ids(self.root / 'train.txt')
        holdout = _land__split_ids(self.root / 'val.txt') | _land__split_ids(self.root / 'test.txt')
        if train & holdout:
            raise ValueError('Author Train and holdout overlap')
        excluded = set(manifest['oem_overlap_excluded_sources'])
        with self.index_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != _land_INDEX_COLUMNS:
                raise ValueError('Hard dry index columns differ')
            self.rows = list(reader)
        identities = [row['id'] for row in self.rows]
        if not identities or len(identities) != len(set(identities)):
            raise ValueError('Hard dry index empty or duplicated')
        self.sources = set()
        for row in self.rows:
            source = row['source']
            if row['id'] not in train or row['id'] in holdout or source in excluded or (not row['id'].startswith(source + '_')):
                raise ValueError(f"Non-Train or excluded hard dry ID: {row['id']}")
            try:
                x, y = (int(row['x']), int(row['y']))
                water, black, fp = (int(row[key]) for key in ('water_pixels', 'black_pixels', 'baseline_fp'))
            except (TypeError, ValueError) as error:
                raise ValueError('Invalid hard dry index number') from error
            if x < 0 or y < 0 or water != 0 or (not 0 <= black <= _land_SIZE * _land_SIZE // 100) or (fp < 1000):
                raise ValueError(f"Invalid hard dry index counts: {row['id']}")
            self.sources.add(source)
        expected_sources = manifest['source_sha256']
        if set(expected_sources) != self.sources:
            raise ValueError('Source hash manifest differs from index')
        for source in self.sources:
            for folder in ('images', 'masks'):
                path = self.root / folder / f'{source}.tif'
                if not path.is_file():
                    raise FileNotFoundError(path)
                if checksum(path) != expected_sources[source][folder]:
                    raise ValueError(f'Source TIFF SHA-256 differs: {path}')

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(index, int) or index < 0 or index >= len(self.rows):
            raise IndexError(index)
        row = self.rows[index]
        source = row['source']
        x, y = (int(row['x']), int(row['y']))
        image_path = self.root / 'images' / f'{source}.tif'
        mask_path = self.root / 'masks' / f'{source}.tif'
        with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
            if image_src.count != 3 or image_src.dtypes != ('uint8',) * 3 or mask_src.count != 1 or (mask_src.dtypes != ('uint8',)) or (image_src.width != mask_src.width) or (image_src.height != mask_src.height) or (x + _land_SIZE > image_src.width) or (y + _land_SIZE > image_src.height):
                raise ValueError(f"Invalid hard dry pair geometry or type: {row['id']}")
            window = Window(x, y, _land_SIZE, _land_SIZE)
            rgb = image_src.read((1, 2, 3), window=window)
            mask = mask_src.read(1, window=window)
        if rgb.shape != (3, _land_SIZE, _land_SIZE) or mask.shape != (_land_SIZE, _land_SIZE) or (not np.all(mask <= 4)):
            raise ValueError(f"Invalid hard dry tile: {row['id']}")
        water_pixels = int(np.count_nonzero(mask == 3))
        black = np.all(rgb == 0, axis=0)
        black_pixels = int(np.count_nonzero(black))
        if water_pixels != int(row['water_pixels']) or water_pixels != 0 or black_pixels != int(row['black_pixels']) or (black_pixels > _land_SIZE * _land_SIZE // 100):
            raise ValueError(f"Hard dry tile differs from index: {row['id']}")
        image = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        water = np.zeros((1, _land_SIZE, _land_SIZE), dtype=np.float32)
        valid = (~black).astype(np.float32)[None, :, :]
        image, water, valid = augment_triplet(image, water, valid)
        return (torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(water)), torch.from_numpy(np.ascontiguousarray(valid)))


# Original: src/xbd_training_data.py :: INDEX_FIELDS
_xbd_INDEX_FIELDS = ['image', 'mask', 'region', 'oem_label', 'water_pixels', 'valid_pixels', 'water_fraction', 'xbd_basename', 'mirror_split', 'mirror_shard', 'mirror_row', 'mirror_revision', 'image_sha256', 'mask_sha256']


# Original: src/xbd_training_data.py :: MIRROR_REVISION
_xbd_MIRROR_REVISION = '079adbab22aa09633dfd50ece0dd0137431a9780'


# Original: src/xbd_training_data.py :: _file_under
def _xbd__file_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path leaves data root: {relative}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


# Original: src/xbd_training_data.py :: load_xbd_train_dataset
def _xbd_load_xbd_train_dataset(root: Path, index_path: Path, expected_sha256: str, author_train_path: Path, forbidden_regions: set[str]) -> WaterDataset:
    root = Path(root).expanduser().resolve()
    index_path = Path(index_path).expanduser().resolve()
    author_train_path = Path(author_train_path).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not author_train_path.is_file():
        raise FileNotFoundError(author_train_path)
    if checksum(index_path) != expected_sha256:
        raise ValueError('xBD training index SHA-256 differs')
    with index_path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _xbd_INDEX_FIELDS:
            raise ValueError('xBD training index fields differ')
        rows = list(reader)
    if not rows:
        raise ValueError('xBD training index is empty')
    author_lines = [line.strip() for line in author_train_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(author_lines) != len(set(author_lines)):
        raise ValueError('Author Train split contains duplicate labels')
    author_train = set(author_lines)
    if not isinstance(forbidden_regions, set):
        raise ValueError('Forbidden regions must be a set')
    seen = set()
    for row in rows:
        region, label = (row['region'], row['oem_label'])
        identity = (row['image'], row['mask'])
        if identity in seen or label not in author_train or region in forbidden_regions:
            raise ValueError(f'Duplicate, non-Train or forbidden xBD pair: {label}')
        seen.add(identity)
        if not region or '/' in region or '..' in region or (not label.startswith(region + '_')) or (not label.endswith('.tif')):
            raise ValueError(f'Invalid xBD region or label: {label}')
        if row['mirror_split'] not in {'train', 'tier3'} or not row['mirror_shard'].startswith(f"data/{row['mirror_split']}-") or row['mirror_revision'] != _xbd_MIRROR_REVISION or (not row['xbd_basename']):
            raise ValueError(f'Non-training xBD source: {label}')
        try:
            if int(row['mirror_row']) < 0:
                raise ValueError('Negative mirror row')
            water_count = int(row['water_pixels'])
            valid_count = int(row['valid_pixels'])
            fraction = float(row['water_fraction'])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid xBD index counts: {label}') from exc
        if not np.isfinite(fraction) or not 0 <= water_count <= valid_count <= 1024 * 1024 or fraction <= 0.05:
            raise ValueError(f'Invalid xBD water fraction: {label}')
        expected_image = f'OpenEarthMap_wo_xBD/{region}/images/{label[:-4]}.png'
        expected_mask = f'OpenEarthMap_wo_xBD/{region}/labels/{label}'
        if row['image'] != expected_image or row['mask'] != expected_mask:
            raise ValueError(f'xBD image/mask path does not match label: {label}')
        image_path = _xbd__file_under(root, row['image'])
        mask_path = _xbd__file_under(root, row['mask'])
        if checksum(image_path) != row['image_sha256'] or checksum(mask_path) != row['mask_sha256']:
            raise ValueError(f'xBD image/mask SHA-256 differs: {label}')
        with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
            if (image_src.width, image_src.height, image_src.count, image_src.dtypes) != (1024, 1024, 3, ('uint8',) * 3) or (mask_src.width, mask_src.height, mask_src.count, mask_src.dtypes) != (1024, 1024, 1, ('uint8',)):
                raise ValueError(f'Invalid xBD pair geometry or dtype: {label}')
            mask = mask_src.read(1)
        if not np.all(mask <= 8):
            raise ValueError(f'Invalid xBD mask class: {label}')
        observed_water = int(np.count_nonzero(mask == 6))
        observed_valid = int(np.count_nonzero((mask >= 1) & (mask <= 8)))
        if observed_water != water_count or observed_valid != valid_count or observed_valid == 0 or (abs(observed_water / observed_valid - fraction) > 1e-12) or (observed_water / observed_valid <= 0.05):
            raise ValueError(f'xBD mask counts differ: {label}')
    return WaterDataset(index_path, root, mode='train')


# Original: src/hard_train_data.py :: REL
_hard_REL = Path('configs/trial_train_hard_20261029_001')


# Original: src/hard_train_data.py :: MANIFEST_SHA
_hard_MANIFEST_SHA = 'df4996a58b2f651f88399c64b7a11ea609bf4f54adcd92d5dab7d852d3e78e6c'


# Original: src/hard_train_data.py :: OEM_FILTERED_SHA
_hard_OEM_FILTERED_SHA = '8c89faaaf76bf0641593576036fb9c86e2ce1177e736fa1d6484f38a76721f2c'


# Original: src/hard_train_data.py :: OEM_VAL_SHA
_hard_OEM_VAL_SHA = '91a8dac45211a4cd43fc14175619492046b312bfd76c5f896b98f1f7b3714709'


# Original: src/hard_train_data.py :: COUNTS
_hard_COUNTS = {'OEM': 1888, 'CAID': 1461, 'LandCover': 69, 'xBD': 54, 'FLAIR': 528}


# Original: src/hard_train_data.py :: FOCUS_COUNTS
_hard_FOCUS_COUNTS = {'OEM': 962, 'CAID': 455, 'LandCover': 1, 'xBD': 54, 'FLAIR': 528}


# Original: src/hard_train_data.py :: CAID_ARCHIVE_SHA
_hard_CAID_ARCHIVE_SHA = '0525f2edbc56624da4b6730cb2eb9365b1aa31740ef5224648bab239c332bd1b'


# Original: src/hard_train_data.py :: LAND_INDEX_SHA
_hard_LAND_INDEX_SHA = '2c4827b84f263e23f71a101567026d68149592fdaa1b6fd74c2869254f304cc2'


# Original: src/hard_train_data.py :: LAND_MANIFEST_SHA
_hard_LAND_MANIFEST_SHA = 'f88f5f7fa75ccb5dbf9901082e7fb72cac0d53f27cf0ee0f591d0fc4d828710f'


# Original: src/hard_train_data.py :: XBD_INDEX_SHA
_hard_XBD_INDEX_SHA = '77d69cfbaa0eddd4244ae9efea6a5ed46223750706a03e948732f9d3f56b6175'


# Original: src/hard_train_data.py :: FLAIR_ARCHIVES
_hard_FLAIR_ARCHIVES = {'D013_2020': ('data/raw/flair_d013/D013_2020.zip', '7fee53e0f7d4194b84b67e3ea8092890c7485918717e10ab624026764ff1cf24'), 'D034_2021': ('data/raw/flair_d034/D034_2021.zip', '9e1c936084c39149d2fcd5516496f3c13acc8b3d4a894ecf0f1a0131859dfb3e')}


# Original: src/hard_train_data.py :: _rows
def _hard__rows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


# Original: src/hard_train_data.py :: _flair_tensor_triplet
def _hard__flair_tensor_triplet(rgb: np.ndarray, label: np.ndarray, rgb_valid: np.ndarray, mask_valid: np.ndarray, *, augment: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rgb.shape != (3, 512, 512) or rgb.dtype != np.uint8 or label.shape != (512, 512) or (label.dtype != np.uint8) or (rgb_valid.shape != label.shape) or (mask_valid.shape != label.shape) or (not np.all(label <= 19)) or np.any(label == 13):
        raise ValueError('FLAIR selected pair geometry, class or pool differs')
    image = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    water = (label == 5).astype(np.float32)[None]
    valid = ((label >= 1) & (label <= 19) & rgb_valid & mask_valid).astype(np.float32)[None]
    if augment:
        image, water, valid = augment_triplet(image, water, valid)
    return (torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(water)), torch.from_numpy(np.ascontiguousarray(valid)))


# Original: src/hard_train_data.py :: FlairSelectedDataset
class _hard_FlairSelectedDataset(Dataset):

    def __init__(self, project: Path, index_path: Path, expected_index_sha: str, archive_sources: dict | None=None):
        self.project = Path(project)
        if checksum(index_path) != expected_index_sha:
            raise ValueError('FLAIR selected index SHA differs')
        self.rows = _hard__rows(index_path)
        if len(self.rows) != 528 or len({(r['source'], r['image']) for r in self.rows}) != 528:
            raise ValueError('FLAIR selected index size/uniqueness differs')
        self.archives = {}
        try:
            for name, (relative, expected_sha) in (archive_sources or _hard_FLAIR_ARCHIVES).items():
                path = self.project / relative
                if checksum(path) != expected_sha:
                    raise ValueError(f'FLAIR archive SHA differs: {name}')
                self.archives[name] = zipfile.ZipFile(path)
            for row in self.rows:
                if row['source'] not in self.archives or 'Z18_UU' in row['image'] or row['image'] not in self.archives[row['source']].namelist() or (row['mask'] not in self.archives[row['source']].namelist()):
                    raise ValueError('FLAIR selected member missing or reserve selected')
        except Exception:
            self.close()
            raise

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        archive = self.archives[row['source']]
        with MemoryFile(archive.read(row['image'])) as mem, mem.open() as image_src:
            if image_src.count != 5 or image_src.dtypes != ('uint8',) * 5 or (image_src.width, image_src.height) != (512, 512):
                raise ValueError('FLAIR RGB format differs')
            rgb = image_src.read((1, 2, 3))
            rgb_valid = np.all(image_src.read_masks((1, 2, 3)) > 0, axis=0)
            crs, transform = (image_src.crs, image_src.transform)
        with MemoryFile(archive.read(row['mask'])) as mem, mem.open() as mask_src:
            known_crs_pair = crs is not None and crs.to_epsg() == 2154 and (mask_src.crs is not None) and str(mask_src.crs).startswith('LOCAL_CS["RGF93 v1 / Lambert-93"')
            if mask_src.count != 1 or mask_src.dtypes != ('uint8',) or (mask_src.width, mask_src.height) != (512, 512) or (not mask_src.transform.almost_equals(transform)) or (mask_src.crs != crs and (not known_crs_pair)):
                raise ValueError('FLAIR mask format or pixel grid differs')
            label = mask_src.read(1)
            mask_valid = mask_src.read_masks(1) > 0
        if int(((label == 5) & rgb_valid & mask_valid).sum()) != int(row['water_pixels']):
            raise ValueError('FLAIR water count differs')
        return _hard__flair_tensor_triplet(rgb, label, rgb_valid, mask_valid, augment=True)

    def close(self):
        for opened in getattr(self, 'archives', {}).values():
            opened.close()
        self.archives = {}

    def __del__(self):
        self.close()


# Original: src/hard_train_data.py :: HardTrainDataset
class _hard_HardTrainDataset(Dataset):

    def __init__(self, rows, sources, indices):
        self.rows, self.sources, self.indices = (rows, sources, indices)
        if len(rows) != len(indices):
            raise ValueError('Train mapping length differs')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source, source_index = self.indices[index]
        return self.sources[source][source_index]

    def close(self):
        for source in self.sources.values():
            if hasattr(source, 'close'):
                source.close()
            elif hasattr(source, 'archive'):
                source.archive.close()


# Original: src/hard_train_data.py :: _portable_archive_hashes
def _hard__portable_archive_hashes(project: Path, config: dict) -> tuple[str, dict]:
    """Accept only source-equivalent subset archives recorded by the transfer builder."""
    path = project / _hard_REL / 'portable_sources.json'
    if not path.exists():
        return (_hard_CAID_ARCHIVE_SHA, _hard_FLAIR_ARCHIVES)
    portable = json.loads(path.read_text(encoding='utf-8'))
    if portable.get('schema_version') != 1 or portable.get('train_manifest_sha256') != config['train_manifest_sha256'] or portable.get('original_caid_sha256') != _hard_CAID_ARCHIVE_SHA or (portable.get('original_flair_sha256') != {name: sha for name, (_, sha) in _hard_FLAIR_ARCHIVES.items()}):
        raise ValueError('Portable source provenance differs')
    caid = portable.get('caid_subset_sha256')
    flair = portable.get('flair_subset_sha256')
    if not isinstance(caid, str) or len(caid) != 64 or (not isinstance(flair, dict)) or (set(flair) != set(_hard_FLAIR_ARCHIVES)) or any((not isinstance(value, str) or len(value) != 64 for value in flair.values())):
        raise ValueError('Portable archive SHA schema differs')
    return (caid, {name: (relative, flair[name]) for name, (relative, _) in _hard_FLAIR_ARCHIVES.items()})


# Original: src/hard_train_data.py :: build_hard_train_dataset
def _hard_build_hard_train_dataset(project: Path, rows: list[dict], config: dict) -> _hard_HardTrainDataset:
    project = Path(project).expanduser().resolve()
    folder = project / _hard_REL
    caid_archive_sha, flair_archives = _hard__portable_archive_hashes(project, config)
    sources = {}
    try:
        sources['OEM'] = WaterDataset(project / config['oem_train'], project / 'data/raw/openearthmap', mode='train')
        sources['CAID'] = _caid_CAIDTrainDataset(project / 'data/raw/caid/CAID.zip', folder / 'caid_selected_index.csv', caid_archive_sha, config['caid_selected_index_sha256'])
        sources['LandCover'] = _land_LandCoverHardDryDataset(project / 'data/raw/landcover/extracted', project / 'configs/landcover_hard_dry_train.csv', project / 'configs/landcover_hard_dry_manifest.json', _hard_LAND_INDEX_SHA, _hard_LAND_MANIFEST_SHA)
        val_regions = {r['region'] for r in _hard__rows(project / config['oem_val'])}
        sources['xBD'] = _xbd_load_xbd_train_dataset(project / 'data/raw/openearthmap', project / 'configs/xbd_extra_train.csv', _hard_XBD_INDEX_SHA, project / 'data/raw/openearthmap/OpenEarthMap_wo_xBD/train.txt', val_regions)
        sources['FLAIR'] = _hard_FlairSelectedDataset(project, folder / 'flair_selected_index.csv', config['flair_selected_index_sha256'], flair_archives)
        mapping = {'OEM': {r['sample_id']: i for i, r in enumerate(sources['OEM'].samples)}, 'CAID': {r['id']: i for i, r in enumerate(sources['CAID'].rows)}, 'LandCover': {r['id']: i for i, r in enumerate(sources['LandCover'].rows)}, 'xBD': {r['oem_label']: i for i, r in enumerate(_hard__rows(project / 'configs/xbd_extra_train.csv'))}, 'FLAIR': {r['source'] + ':' + r['image']: i for i, r in enumerate(sources['FLAIR'].rows)}}
        indices = []
        for row in rows:
            source, identity = (row['source'], row['pair_id'])
            if identity not in mapping[source]:
                raise ValueError(f'Train pair missing from verified source: {source}/{identity}')
            indices.append((source, mapping[source][identity]))
        return _hard_HardTrainDataset(rows, sources, indices)
    except Exception:
        for source in sources.values():
            if hasattr(source, 'close'):
                source.close()
        raise


# Original: src/glh_coast_train_data.py :: REL
_glh_REL = Path('configs/trial_train_glh_coast_20261101_001')


# Original: src/glh_coast_train_data.py :: DATA_REL
_glh_DATA_REL = Path('data/processed/glh_coast_train_20261101_001')


# Original: src/glh_coast_train_data.py :: CONFIG_SHA
_glh_CONFIG_SHA = '2b69a468a3b5f09ff6cda612b23057df4e41a7440aa1dab9e7a26f102c8f1d6e'


# Original: src/glh_coast_train_data.py :: TRAIN_SHA
_glh_TRAIN_SHA = '392cdbd3692e4369c1d887305c773dfb8d321d9f32f486dbd9f292c1fa498106'


# Original: src/glh_coast_train_data.py :: BASE_SHA
_glh_BASE_SHA = '25050dafa70461a43e202d3930adb988be954bf6d2ec6f84582cf2f00373cc17'


# Original: src/glh_coast_train_data.py :: FILES_SHA
_glh_FILES_SHA = '5696035bc6f38c2b49dc47c8a718e5e64a1aba95d2382002e0bea352cd3b4763'


# Original: src/glh_coast_train_data.py :: VAL_SHA
_glh_VAL_SHA = '5c285e82048e9733ee8e8428ad743a850beba7ac6ce6e9611b539018db56ce3c'


# Original: src/glh_coast_train_data.py :: COUNTS
_glh_COUNTS = {'GLH-Water': 700, 'CoastTrain_NAIP11': 700}


# Original: src/glh_coast_train_data.py :: STAGES
_glh_STAGES = {('GLH-Water', 'visual_type'): 393, ('GLH-Water', 'frozen_model_error'): 307, ('CoastTrain_NAIP11', 'visual_type'): 380, ('CoastTrain_NAIP11', 'frozen_model_error'): 320}


# Original: src/glh_coast_train_data.py :: _sha
def _glh__sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# Original: src/glh_coast_train_data.py :: _read
def _glh__read(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


# Original: src/glh_coast_train_data.py :: _source_path
def _glh__source_path(project: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError('External source path must be relative')
    path = (project / relative).resolve()
    if not path.is_relative_to(project / _glh_DATA_REL) or not path.is_file():
        raise ValueError('External source path outside frozen data or missing')
    return path


# Original: src/glh_coast_train_data.py :: load_mixed_train_rows
def _glh_load_mixed_train_rows(project: Path) -> tuple[list[dict], list[dict], dict]:
    """Verify frozen manifest, base identity and every external file byte hash."""
    project = Path(project).expanduser().resolve()
    folder = project / _glh_REL
    if _glh__sha(folder / 'manifest.json') != _glh_CONFIG_SHA:
        raise ValueError('Mixed Train config SHA differs')
    config = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    required = {'schema_version': 1, 'train_manifest': str(_glh_REL / 'train_manifest.csv'), 'train_manifest_sha256': _glh_TRAIN_SHA, 'base_manifest': str(_glh_REL / 'base_4023/train_manifest.csv'), 'base_manifest_sha256': _glh_BASE_SHA, 'external_file_index': str(_glh_REL / 'external_files_sha256.csv'), 'external_file_index_sha256': _glh_FILES_SHA, 'oem_val': 'configs/trial_train_tyrolw_20261101_001/oem_val.csv', 'oem_val_sha256': _glh_VAL_SHA, 'train_unique_pairs': 5423, 'base_pairs': 4023, 'glh_pairs': 700, 'coast_pairs': 700, 'materialized_data_root': str(_glh_DATA_REL), 'materialized_data_files': 3500, 'source_epoch': 29, 'source_optimizer_step': 468667, 'semantic_review_complete': False, 'glh_explicit_file_rights_confirmed': False}
    if any((config.get(key) != value for key, value in required.items())):
        raise ValueError('Mixed Train schema or identity differs')
    for relative, digest in ((config['train_manifest'], _glh_TRAIN_SHA), (config['base_manifest'], _glh_BASE_SHA), (config['external_file_index'], _glh_FILES_SHA), (config['oem_val'], _glh_VAL_SHA)):
        if _glh__sha(project / relative) != digest:
            raise ValueError(f'Frozen input SHA differs: {relative}')
    rows = _glh__read(project / config['train_manifest'])
    base = _glh__read(project / config['base_manifest'])
    index = _glh__read(project / config['external_file_index'])
    if len(rows) != 5423 or len(base) != 4023 or len(index) != 1400 or (len({(r['source'], r['pair_id']) for r in rows}) != 5423) or any(({key: row[key] for key in original} != original for row, original in zip(rows[:4023], base, strict=True))):
        raise ValueError('Mixed Train base/count/uniqueness differs')
    external = rows[4023:]
    if Counter((r['source'] for r in external)) != _glh_COUNTS or Counter(((r['source'], r['selection_stage']) for r in external)) != _glh_STAGES or [(r['source'], r['pair_id']) for r in index] != [(r['source'], r['pair_id']) for r in external]:
        raise ValueError('External source/stage/file-index identity differs')
    for row, file_row in zip(external, index, strict=True):
        source, pair_id = (row['source'], row['pair_id'])
        if row['candidate_status'] != 'candidate_only_not_training_ready' or row['old_route_role'] != 'new_candidate_not_admitted' or (source == 'GLH-Water' and (row['valid_mask'] or pair_id.startswith(('GLH:54:', 'GLH:64:', 'GLH:98:')))) or (source == 'CoastTrain_NAIP11' and (not row['valid_mask'])):
            raise ValueError(f'External source semantics differ: {pair_id}')
        for field in ('image', 'mask', 'valid_mask'):
            relative = row[field]
            expected = file_row[field + '_sha256']
            if not relative:
                if expected:
                    raise ValueError('Unexpected valid file hash')
                continue
            if len(expected) != 64 or _glh__sha(_glh__source_path(project, relative)) != expected:
                raise ValueError(f'External file SHA differs: {pair_id}/{field}')
    return (base, external, config)


# Original: src/glh_coast_train_data.py :: ExternalPairDataset
class _glh_ExternalPairDataset(Dataset):
    """Binary 512-pixel pair with an explicit valid mask and paired augmentation."""

    def __init__(self, project: Path, rows: list[dict], *, augment: bool=True):
        self.project, self.rows, self.augment = (Path(project), rows, augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        if row['source'] not in _glh_COUNTS or (row['source'] == 'GLH-Water') != (not row['valid_mask']):
            raise ValueError('External source/valid-mask contract differs')
        paths = [self.project / row[name] for name in ('image', 'mask')]
        with Image.open(paths[0]) as image_file, Image.open(paths[1]) as mask_file:
            if image_file.mode != 'RGB' or mask_file.mode != 'L' or image_file.size != (512, 512) or (mask_file.size != (512, 512)):
                raise ValueError('External RGB/mask geometry or mode differs')
            rgb = np.asarray(image_file)
            mask = np.asarray(mask_file)
        if row['valid_mask']:
            with Image.open(self.project / row['valid_mask']) as valid_file:
                if valid_file.mode != 'L' or valid_file.size != (512, 512):
                    raise ValueError('External valid-mask geometry or mode differs')
                valid = np.asarray(valid_file)
        else:
            valid = np.full((512, 512), 255, dtype=np.uint8)
        if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8 or mask.dtype != np.uint8 or (valid.dtype != np.uint8) or (not np.isin(mask, (0, 255)).all()) or (not np.isin(valid, (0, 255)).all()) or np.any((mask == 255) & (valid == 0)) or (int((valid == 255).sum()) != int(row['valid_pixels'])) or (int((mask == 255).sum()) != int(row['water_pixels'])):
            raise ValueError('External water/valid values or counts differ')
        image = (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - MEAN) / STD
        water = (mask == 255).astype(np.float32)[None]
        known = (valid == 255).astype(np.float32)[None]
        if self.augment:
            image, water, known = augment_triplet(image, water, known)
        return (torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(water)), torch.from_numpy(np.ascontiguousarray(known)))


# Original: src/glh_coast5406_train_data.py :: REL
_glh5406_REL = Path('configs/trial_train_glh_coast_owner_corrected_20261101_003')


# Original: src/glh_coast5406_train_data.py :: MANIFEST_SHA
_glh5406_MANIFEST_SHA = '5c92191cd1877149d14e4b60838c5b096a71820bdfd0b802730204dba86267d5'


# Original: src/glh_coast5406_train_data.py :: TRAIN_SHA
_glh5406_TRAIN_SHA = '460833ec51f30d706c991b9b330f569f32d89f1a9266d42197da976e953c6e9f'


# Original: src/glh_coast5406_train_data.py :: EXTERNAL_SHA
_glh5406_EXTERNAL_SHA = '4174e43df0a43a21bd02929c40b62af0bcd188762b557e5801297e1f5704a1be'


# Original: src/glh_coast5406_train_data.py :: QUARANTINE_SHA
_glh5406_QUARANTINE_SHA = 'a9140fe1d4a819ac3738f2d9a7ff3a53b294bd06dd2eec831c782cb733c82fa9'


# Original: src/glh_coast5406_train_data.py :: OWNER_SHA
_glh5406_OWNER_SHA = '421a86157ace8d6dd207855cdf3c99272c4ffee33e69e08bd37839b5df5ccf0e'


# Original: src/glh_coast5406_train_data.py :: PARENT_SHA
_glh5406_PARENT_SHA = '4e346a4df43fd4fd11848c43bf7b7c67a6287be488ec0e9c71d38545f1e6be0e'


# Original: src/glh_coast5406_train_data.py :: COUNTS
_glh5406_COUNTS = {'GLH-Water': 688, 'CoastTrain_NAIP11': 695}


# Original: src/glh_coast5406_train_data.py :: STAGES
_glh5406_STAGES = {('GLH-Water', 'visual_type'): 387, ('GLH-Water', 'frozen_model_error'): 301, ('CoastTrain_NAIP11', 'visual_type'): 376, ('CoastTrain_NAIP11', 'frozen_model_error'): 319}


# Original: src/glh_coast5406_train_data.py :: EXTRA_EXCLUDED
_glh5406_EXTRA_EXCLUDED = {'GLH:27:4608:2048', 'GLH:27:7680:1536'}


# Original: src/glh_coast5406_train_data.py :: _read
def _glh5406__read(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


# Original: src/glh_coast5406_train_data.py :: load_screened_train_rows
def _glh5406_load_screened_train_rows(project: Path) -> tuple[list[dict], list[dict], dict]:
    """Verify the exact 5423-minus-17 subset and source file identities."""
    project = Path(project).expanduser().resolve()
    folder = project / _glh5406_REL
    hashes = {'manifest.json': _glh5406_MANIFEST_SHA, 'train_manifest.csv': _glh5406_TRAIN_SHA, 'external_files_sha256.csv': _glh5406_EXTERNAL_SHA, 'owner_decisions.csv': _glh5406_OWNER_SHA, 'quarantine.csv': _glh5406_QUARANTINE_SHA}
    if any((_glh__sha(folder / name) != expected for name, expected in hashes.items())):
        raise ValueError('Screened Train file SHA differs')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    required = {'schema_version': 1, 'train_sha256': _glh5406_TRAIN_SHA, 'external_file_index_sha256': _glh5406_EXTERNAL_SHA, 'owner_decisions_sha256': _glh5406_OWNER_SHA, 'quarantine_sha256': _glh5406_QUARANTINE_SHA, 'parent_sha256': _glh5406_PARENT_SHA, 'base_pairs': 4023, 'new_glh_pairs': 688, 'new_coast_pairs': 695, 'train_pairs': 5406, 'external_index_pairs': 1383, 'quarantine_pairs': 17, 'training_performed': False}
    if any((manifest.get(key) != value for key, value in required.items())) or manifest.get('train_manifest') != str(_glh5406_REL / 'train_manifest.csv'):
        raise ValueError('Screened Train metadata differs')
    base, old_external, old_config = _glh_load_mixed_train_rows(project)
    old_rows = _glh5406__read(project / old_config['train_manifest'])
    selected = _glh5406__read(folder / 'train_manifest.csv')
    indexed = _glh5406__read(folder / 'external_files_sha256.csv')
    quarantine = _glh5406__read(folder / 'quarantine.csv')
    excluded = {(r['source'], r['pair_id']) for r in quarantine}
    if len(quarantine) != 17 or len(excluded) != 17 or len(selected) != 5406 or (len(indexed) != 1383) or (len({(r['source'], r['pair_id']) for r in selected}) != 5406) or (not _glh5406_EXTRA_EXCLUDED.issubset({r['pair_id'] for r in quarantine})) or any((r['decision'] not in {'exclude_owner', 'quarantine_unresolved', 'exclude_semantic_conflict'} for r in quarantine)):
        raise ValueError('Screened membership or quarantine differs')
    expected_selected = [r for r in old_rows if (r['source'], r['pair_id']) not in excluded]
    if selected != expected_selected or [{key: row[key] for key in original} for row, original in zip(selected[:4023], base, strict=True)] != base or Counter((r['source'] for r in selected[4023:])) != _glh5406_COUNTS or (Counter(((r['source'], r['selection_stage']) for r in selected[4023:])) != _glh5406_STAGES):
        raise ValueError('Screened rows differ from exact original subset')
    old_index = _glh5406__read(project / old_config['external_file_index'])
    expected_index = [r for r in old_index if (r['source'], r['pair_id']) not in excluded]
    if indexed != expected_index or [(r['source'], r['pair_id']) for r in indexed] != [(r['source'], r['pair_id']) for r in selected[4023:]]:
        raise ValueError('Screened external SHA index differs')
    config = {**manifest, 'train_manifest_sha256': _glh5406_TRAIN_SHA, 'train_unique_pairs': 5406, 'oem_val': old_config['oem_val'], 'oem_val_sha256': old_config['oem_val_sha256'], 'external_file_index': str(_glh5406_REL / 'external_files_sha256.csv')}
    return (base, selected[4023:], config)


# Original: src/glh_coast5406_train_data.py :: ScreenedMixedTrainDataset
class _glh5406_ScreenedMixedTrainDataset(Dataset):
    """The original 4023 rows followed by exactly 1383 selected PNG pairs."""

    def __init__(self, base: Dataset, external: _glh_ExternalPairDataset, rows: list[dict]):
        if len(base) != 4023 or len(external) != 1383 or len(rows) != 5406:
            raise ValueError('Screened dataset length differs')
        self.base, self.external, self.rows = (base, external, rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.base[index] if index < 4023 else self.external[index - 4023]

    def close(self) -> None:
        self.base.close()


# Original: src/glh_coast5406_train_data.py :: build_screened_train_dataset
def _glh5406_build_screened_train_dataset(project: Path, base_rows: list[dict], external_rows: list[dict], config: dict) -> _glh5406_ScreenedMixedTrainDataset:
    """Keep the old source readers and append only the approved external rows."""
    project = Path(project).expanduser().resolve()
    old_rows, old_config = _tyrol_load_revision(project, project / 'configs/trial_train_tyrolw_20261101_001')
    if len(base_rows) != 4023 or [{**r, 'focus_hard': int(r['focus_hard'])} for r in base_rows] != old_rows or len(external_rows) != 1383 or (config.get('train_manifest_sha256') != _glh5406_TRAIN_SHA):
        raise ValueError('Screened old/external membership differs')
    base = _hard_build_hard_train_dataset(project, old_rows, old_config)
    try:
        return _glh5406_ScreenedMixedTrainDataset(base, _glh_ExternalPairDataset(project, external_rows), old_rows + external_rows)
    except Exception:
        base.close()
        raise


# Original: src/base_train_768.py :: CONTEXT_SIZE
_base768_CONTEXT_SIZE = 768


# Original: src/base_train_768.py :: LABEL_SIZE
_base768_LABEL_SIZE = 512


# Original: src/base_train_768.py :: _context_origin
def _base768__context_origin(tile_origin: int, source_length: int) -> int:
    """Keep a selected 512 tile inside a real 768 source window."""
    return min(max(0, tile_origin - (_base768_CONTEXT_SIZE - _base768_LABEL_SIZE) // 2), source_length - _base768_CONTEXT_SIZE)


# Original: src/base_train_768.py :: _tensor_triplet
def _base768__tensor_triplet(rgb: np.ndarray, water_tile: np.ndarray, valid_tile: np.ndarray, tile_x: int, tile_y: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rgb.shape != (3, _base768_CONTEXT_SIZE, _base768_CONTEXT_SIZE) or rgb.dtype != np.uint8 or water_tile.shape != (_base768_LABEL_SIZE, _base768_LABEL_SIZE) or (valid_tile.shape != (_base768_LABEL_SIZE, _base768_LABEL_SIZE)) or (not 0 <= tile_x <= _base768_CONTEXT_SIZE - _base768_LABEL_SIZE) or (not 0 <= tile_y <= _base768_CONTEXT_SIZE - _base768_LABEL_SIZE):
        raise ValueError('Native RGB context or supervised tile shape differs')
    image = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    water = np.zeros((1, _base768_CONTEXT_SIZE, _base768_CONTEXT_SIZE), dtype=np.float32)
    valid = np.zeros_like(water)
    region = (slice(tile_y, tile_y + _base768_LABEL_SIZE), slice(tile_x, tile_x + _base768_LABEL_SIZE))
    water[0, *region] = water_tile.astype(np.float32)
    valid[0, *region] = valid_tile.astype(np.float32)
    image, water, valid = augment_triplet(image, water, valid)
    return (torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(water)), torch.from_numpy(np.ascontiguousarray(valid)))


# Original: src/base_train_768.py :: _read_oem_768
def _base768__read_oem_768(source, index: int):
    """Keep OEM/xBD 512 sampling and labels, adding source RGB around them."""
    sample = source.samples[index]
    image_path = Path(source.data_dir) / sample['image']
    mask_path = Path(source.data_dir) / sample['mask']
    with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
        width, height = (image_src.width, image_src.height)
        if image_src.count != 3 or mask_src.count != 1 or mask_src.dtypes != ('uint8',) or ((mask_src.width, mask_src.height) != (width, height)) or (width < _base768_CONTEXT_SIZE) or (height < _base768_CONTEXT_SIZE):
            raise ValueError(f'Invalid native OEM/xBD pair: {image_path}')
        if image_src.crs is not None and mask_src.crs is not None and (image_src.crs != mask_src.crs or not image_src.transform.almost_equals(mask_src.transform)):
            raise ValueError(f'OEM/xBD image and mask grids differ: {image_path}')
        x, y = source._random_position(mask_src)
        if not 0 <= x <= width - _base768_LABEL_SIZE or not 0 <= y <= height - _base768_LABEL_SIZE:
            raise ValueError(f'Invalid selected OEM/xBD tile: {image_path}')
        context_x = _base768__context_origin(x, width)
        context_y = _base768__context_origin(y, height)
        rgb_window = Window(context_x, context_y, _base768_CONTEXT_SIZE, _base768_CONTEXT_SIZE)
        tile_window = Window(x, y, _base768_LABEL_SIZE, _base768_LABEL_SIZE)
        rgb = image_src.read((1, 2, 3), window=rgb_window)
        label = mask_src.read(1, window=tile_window)
        rgb_valid = np.all(image_src.read_masks((1, 2, 3), window=tile_window) > 0, axis=0)
        label_valid = mask_src.read_masks(1, window=tile_window) > 0
    if rgb.dtype != np.uint8 or label.dtype != np.uint8 or label.shape != (_base768_LABEL_SIZE, _base768_LABEL_SIZE) or (not np.all(label <= 8)):
        raise ValueError(f'Invalid native OEM/xBD data: {image_path}')
    rgb = source._augment_rgb(rgb)
    water_tile = label == WATER_CLASS
    valid_tile = (label >= 1) & (label <= 8) & rgb_valid & label_valid
    return _base768__tensor_triplet(rgb, water_tile, valid_tile, x - context_x, y - context_y)


# Original: src/base_train_768.py :: _read_landcover_768
def _base768__read_landcover_768(source, index: int):
    """Place the admitted dry 512 tile inside true source RGB context."""
    row = source.rows[index]
    name = row['source']
    x, y = (int(row['x']), int(row['y']))
    image_path = Path(source.root) / 'images' / f'{name}.tif'
    mask_path = Path(source.root) / 'masks' / f'{name}.tif'
    with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
        width, height = (image_src.width, image_src.height)
        if image_src.count != 3 or image_src.dtypes != ('uint8',) * 3 or mask_src.count != 1 or (mask_src.dtypes != ('uint8',)) or ((mask_src.width, mask_src.height) != (width, height)) or (width < _base768_CONTEXT_SIZE) or (height < _base768_CONTEXT_SIZE) or (x < 0) or (y < 0) or (x + _base768_LABEL_SIZE > width) or (y + _base768_LABEL_SIZE > height):
            raise ValueError(f"Invalid LandCover pair: {row['id']}")
        context_x = _base768__context_origin(x, width)
        context_y = _base768__context_origin(y, height)
        rgb = image_src.read((1, 2, 3), window=Window(context_x, context_y, _base768_CONTEXT_SIZE, _base768_CONTEXT_SIZE))
        label = mask_src.read(1, window=Window(x, y, _base768_LABEL_SIZE, _base768_LABEL_SIZE))
    if rgb.shape != (3, _base768_CONTEXT_SIZE, _base768_CONTEXT_SIZE) or label.shape != (_base768_LABEL_SIZE, _base768_LABEL_SIZE) or (not np.all(label <= 4)):
        raise ValueError(f"Invalid LandCover tile: {row['id']}")
    tile_x, tile_y = (x - context_x, y - context_y)
    tile_rgb = rgb[:, tile_y:tile_y + _base768_LABEL_SIZE, tile_x:tile_x + _base768_LABEL_SIZE]
    black = np.all(tile_rgb == 0, axis=0)
    if int(np.count_nonzero(label == 3)) != int(row['water_pixels']) or int(row['water_pixels']) != 0 or int(np.count_nonzero(black)) != int(row['black_pixels']):
        raise ValueError(f"LandCover indexed tile differs: {row['id']}")
    return _base768__tensor_triplet(rgb, np.zeros_like(label, dtype=np.float32), ~black, tile_x, tile_y)


# Original: src/base_train_768.py :: BaseMixed768Dataset
class _base768_BaseMixed768Dataset(Dataset):
    """Reuse frozen 4023 pair membership; enlarge only available Train RGB."""

    def __init__(self, base):
        if len(base.rows) != len(base.indices) or len(base.rows) != len(base):
            raise ValueError('Base Train rows and source mapping differ')
        self.base = base
        self.rows = base.rows
        self.indices = base.indices
        self.sources = base.sources
        self.context_sizes = []
        native = Counter()
        fallback = Counter()
        geometry_cache = {}
        for source_name, source_index in self.indices:
            if source_name not in base.sources:
                raise ValueError(f'Unknown base source: {source_name}')
            if source_name in {'OEM', 'xBD'}:
                source = base.sources[source_name]
                if source_index < 0 or source_index >= len(source.samples):
                    raise ValueError(f'Unknown {source_name} source index')
                row = source.samples[source_index]
                image_path = Path(source.data_dir) / row['image']
                mask_path = Path(source.data_dir) / row['mask']
            elif source_name == 'LandCover':
                source = base.sources[source_name]
                if source_index < 0 or source_index >= len(source.rows):
                    raise ValueError('Unknown LandCover source index')
                name = source.rows[source_index]['source']
                image_path = Path(source.root) / 'images' / f'{name}.tif'
                mask_path = Path(source.root) / 'masks' / f'{name}.tif'
            elif source_name in {'CAID', 'FLAIR'}:
                self.context_sizes.append(_base768_LABEL_SIZE)
                fallback[source_name] += 1
                continue
            else:
                raise ValueError(f'Unexpected base source: {source_name}')
            cache_key = (image_path, mask_path)
            if cache_key not in geometry_cache:
                with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
                    if (image_src.width, image_src.height) != (mask_src.width, mask_src.height):
                        raise ValueError(f'Base RGB/mask geometry differs: {image_path}')
                    geometry_cache[cache_key] = (image_src.width, image_src.height)
            width, height = geometry_cache[cache_key]
            use_native = width >= _base768_CONTEXT_SIZE and height >= _base768_CONTEXT_SIZE
            if source_name == 'LandCover' and (not use_native):
                raise ValueError(f'LandCover source has no native context: {image_path}')
            size = _base768_CONTEXT_SIZE if use_native else _base768_LABEL_SIZE
            self.context_sizes.append(size)
            (native if use_native else fallback)[source_name] += 1
        self.native768_counts = dict(native)
        self.fallback512_counts = dict(fallback)
        self._closed = False

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise IndexError(index)
        if self.context_sizes[index] == _base768_LABEL_SIZE:
            return self.base[index]
        source_name, source_index = self.indices[index]
        source = self.base.sources[source_name]
        if source_name in {'OEM', 'xBD'}:
            return _base768__read_oem_768(source, source_index)
        if source_name == 'LandCover':
            return _base768__read_landcover_768(source, source_index)
        raise ValueError(f'Native 768 source is not supported: {source_name}')

    def close(self):
        if not self._closed:
            self.base.close()
            self._closed = True


# Original: src/base_train_768.py :: build_base_mixed_768_dataset
def _base768_build_base_mixed_768_dataset(base) -> _base768_BaseMixed768Dataset:
    """Wrap verified base Train without modifying its membership or source readers."""
    return _base768_BaseMixed768Dataset(base)


# Original: src/external_train_768.py :: DEFAULT_RAW
_ext768_DEFAULT_RAW = Path('outputs/checks/glh_coast_intake_20261101_001')


# Original: src/external_train_768.py :: DEFAULT_DEST
_ext768_DEFAULT_DEST = Path('data/processed/glh_coast_context768_20261102_001')


# Original: src/external_train_768.py :: INDEX_FIELDS
_ext768_INDEX_FIELDS = ('index', 'source', 'pair_id', 'mode', 'reason', 'context_image', 'context_sha256', 'source_rgb', 'source_rgb_sha256', 'context_x', 'context_y', 'offset_x', 'offset_y', 'supervision')


# Original: src/external_train_768.py :: _sha
def _ext768__sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# Original: src/external_train_768.py :: plan_context_window
def _ext768_plan_context_window(x: int, y: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Return 768 crop origin and the old 512 tile's offset within it."""
    if any((not isinstance(v, int) for v in (x, y, width, height))) or width < 512 or height < 512 or (x < 0) or (y < 0) or (x + 512 > width) or (y + 512 > height):
        raise ValueError('Old 512 window is outside raw RGB')
    if width < 768 or height < 768:
        return None
    context_x = max(0, min(x - 128, width - 768))
    context_y = max(0, min(y - 128, height - 768))
    offset_x, offset_y = (x - context_x, y - context_y)
    if not (0 <= offset_x <= 256 and 0 <= offset_y <= 256):
        raise AssertionError('Old tile must fit completely inside context')
    return (context_x, context_y, offset_x, offset_y)


# Original: src/external_train_768.py :: _raw_path
def _ext768__raw_path(source_root: Path, row: dict) -> tuple[Path, int, int]:
    pair_id = row['pair_id']
    if row['source'] == 'GLH-Water':
        parts = pair_id.split(':')
        if len(parts) != 4 or parts[0] != 'GLH' or (not parts[1].isdigit()):
            raise ValueError(f'Invalid GLH pair ID: {pair_id}')
        return (source_root / 'raw_type_screen' / f'{int(parts[1]):02d}.jpg', int(parts[2]), int(parts[3]))
    if row['source'] == 'CoastTrain_NAIP11':
        parts = pair_id.rsplit(':', 2)
        if len(parts) != 3 or not parts[0].startswith('Coast:'):
            raise ValueError(f'Invalid Coast pair ID: {pair_id}')
        name = parts[0][len('Coast:'):]
        if Path(name).name != name or not name.endswith('.npz'):
            raise ValueError(f'Invalid Coast source name: {name}')
        return (source_root / 'naip11_type_pool' / name, int(parts[1]), int(parts[2]))
    raise ValueError(f"Unexpected external source: {row['source']}")


# Original: src/external_train_768.py :: _read_selected
def _ext768__read_selected(project: Path, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the frozen 512 pair; reject malformed labels instead of repairing."""
    with Image.open(project / row['image']) as file:
        if file.mode != 'RGB' or file.size != (512, 512):
            raise ValueError(f"Old RGB differs: {row['pair_id']}")
        rgb = np.asarray(file)
    with Image.open(project / row['mask']) as file:
        if file.mode != 'L' or file.size != (512, 512):
            raise ValueError(f"Old water mask differs: {row['pair_id']}")
        water = np.asarray(file)
    if row['valid_mask']:
        with Image.open(project / row['valid_mask']) as file:
            if file.mode != 'L' or file.size != (512, 512):
                raise ValueError(f"Old valid mask differs: {row['pair_id']}")
            valid = np.asarray(file)
    else:
        valid = np.full((512, 512), 255, dtype=np.uint8)
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8 or water.dtype != np.uint8 or (valid.dtype != np.uint8) or (not np.isin(water, (0, 255)).all()) or (not np.isin(valid, (0, 255)).all()) or np.any((water == 255) & (valid == 0)) or (int((water == 255).sum()) != int(row['water_pixels'])) or (int((valid == 255).sum()) != int(row['valid_pixels'])):
        raise ValueError(f"Old water/valid contract differs: {row['pair_id']}")
    return (rgb, water, valid)


# Original: src/external_train_768.py :: _load_raw_rgb
def _ext768__load_raw_rgb(path: Path, source: str) -> np.ndarray | Image.Image:
    if source == 'GLH-Water':
        with Image.open(path) as file:
            if file.mode != 'RGB':
                raise ValueError('GLH raw RGB mode differs')
            return file.copy()
    with np.load(path, allow_pickle=False) as archive:
        if 'orig_image' not in archive.files:
            raise ValueError('Coast orig_image missing')
        rgb = archive['orig_image']
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Coast orig_image RGB geometry or dtype differs')
    return rgb


# Original: src/external_train_768.py :: materialize_external_768_context
def _ext768_materialize_external_768_context(project: Path, rows: list[dict], destination: Path, *, source_root: Path | None=None) -> Path:
    """Write a non-overwriting 768 context catalog with explicit 512 fallbacks."""
    project = Path(project).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    source_root = (project / _ext768_DEFAULT_RAW if source_root is None else Path(source_root)).expanduser().resolve()
    if not destination.is_relative_to(project):
        raise ValueError('Context destination must be inside project')
    if destination.exists():
        raise ValueError('Context destination already exists; no overwrite')
    if not rows or len({(r['source'], r['pair_id']) for r in rows}) != len(rows):
        raise ValueError('External rows are empty or duplicate')
    for row in rows:
        _ext768__read_selected(project, row)
    destination.mkdir(parents=True)
    grouped: dict[Path, list[tuple[int, dict, int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        path, x, y = _ext768__raw_path(source_root, row)
        grouped[path].append((index, row, x, y))
    records: list[dict | None] = [None] * len(rows)
    for path, entries in grouped.items():
        source = entries[0][1]['source']
        source_relative = str(path.relative_to(project)) if path.is_relative_to(project) else str(path)
        common = {'source_rgb': source_relative}
        if not path.is_file():
            reason, source_hash, raw = ('raw_missing', '', None)
        else:
            source_hash = _ext768__sha(path)
            if any((row['source_image_sha256'] != source_hash for _, row, _, _ in entries)):
                reason, raw = ('raw_sha_mismatch', None)
            else:
                try:
                    raw = _ext768__load_raw_rgb(path, source)
                    reason = ''
                except (OSError, ValueError, KeyError) as exc:
                    raw, reason = (None, 'raw_unreadable')
        common['source_rgb_sha256'] = source_hash
        for index, row, x, y in entries:
            record = {name: '' for name in _ext768_INDEX_FIELDS}
            record.update(common)
            record.update({'index': index, 'source': source, 'pair_id': row['pair_id'], 'supervision': 'selected_512_only'})
            fallback_reason = reason
            if raw is not None:
                width, height = raw.size if isinstance(raw, Image.Image) else (raw.shape[1], raw.shape[0])
                try:
                    plan = _ext768_plan_context_window(x, y, width, height)
                except ValueError:
                    plan = None
                    fallback_reason = 'bad_coordinates'
                if plan is None and (not fallback_reason):
                    fallback_reason = 'raw_smaller_than_768'
                if plan is not None:
                    cx, cy, ox, oy = plan
                    tile = np.asarray(raw.crop((cx, cy, cx + 768, cy + 768))) if isinstance(raw, Image.Image) else raw[cy:cy + 768, cx:cx + 768]
                    selected_rgb, _, _ = _ext768__read_selected(project, row)
                    if tile.shape != (768, 768, 3) or not np.array_equal(tile[oy:oy + 512, ox:ox + 512], selected_rgb):
                        fallback_reason = 'center_rgb_mismatch'
                    else:
                        target = destination / f'context_{index:04d}.png'
                        Image.fromarray(np.asarray(tile, dtype=np.uint8)).save(target)
                        record.update({'mode': '768', 'reason': '', 'context_image': str(target.relative_to(project)), 'context_sha256': _ext768__sha(target), 'context_x': cx, 'context_y': cy, 'offset_x': ox, 'offset_y': oy})
            if not record['mode']:
                record.update({'mode': '512', 'reason': fallback_reason})
            records[index] = record
        if isinstance(raw, Image.Image):
            raw.close()
    if any((item is None for item in records)):
        raise AssertionError('Context index omitted a row')
    index_path = destination / 'index.csv'
    with index_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=_ext768_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    counts = Counter(((r['source'], r['mode'], r['reason']) for r in records))
    summary = {'rows': len(records), 'index_sha256': _ext768__sha(index_path), 'source_counts': {f'{source}|{mode}|{reason}': count for (source, mode, reason), count in sorted(counts.items())}, 'supervision': 'selected_512_only', 'training_performed': False}
    (destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return index_path


# Original: src/external_train_768.py :: ExternalContextDataset
class _ext768_ExternalContextDataset(Dataset):
    """Read 768 RGB with the frozen 512 labels inserted, or exact old 512."""

    def __init__(self, project: Path, rows: list[dict], index_file: Path, *, augment: bool=True):
        self.project = Path(project).expanduser().resolve()
        self.rows, self.augment = (rows, augment)
        self.fallback = _glh_ExternalPairDataset(self.project, rows, augment=augment)
        index_file = Path(index_file).expanduser().resolve()
        with index_file.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _ext768_INDEX_FIELDS:
                raise ValueError('Context index schema differs')
            self.index = list(reader)
        if len(self.index) != len(rows):
            raise ValueError('Context index length differs')
        self.context_paths: list[Path | None] = []
        for number, (item, row) in enumerate(zip(self.index, rows, strict=True)):
            if (item['index'], item['source'], item['pair_id'], item['supervision']) != (str(number), row['source'], row['pair_id'], 'selected_512_only'):
                raise ValueError('Context index identity differs')
            if item['mode'] == '512':
                if not item['reason'] or item['context_image']:
                    raise ValueError('Context fallback reason/path differs')
                self.context_paths.append(None)
            elif item['mode'] == '768':
                if item['source_rgb_sha256'] != row['source_image_sha256']:
                    raise ValueError('Context source RGB SHA differs')
                relative = Path(item['context_image'])
                if relative.is_absolute():
                    raise ValueError('Context image path must be relative')
                path = (self.project / relative).resolve()
                if not path.is_relative_to(index_file.parent) or not path.is_file() or _ext768__sha(path) != item['context_sha256']:
                    raise ValueError('Context image SHA/path differs')
                ox, oy = (int(item['offset_x']), int(item['offset_y']))
                if item['reason'] or not (0 <= ox <= 256 and 0 <= oy <= 256):
                    raise ValueError('Context mode or offset differs')
                self.context_paths.append(path)
            else:
                raise ValueError('Unknown context mode')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        path = self.context_paths[index]
        if path is None:
            return self.fallback[index]
        row, item = (self.rows[index], self.index[index])
        with Image.open(path) as file:
            if file.mode != 'RGB' or file.size != (768, 768):
                raise ValueError('Context RGB geometry differs')
            rgb = np.asarray(file)
        selected_rgb, old_water, old_valid = _ext768__read_selected(self.project, row)
        ox, oy = (int(item['offset_x']), int(item['offset_y']))
        if not np.array_equal(rgb[oy:oy + 512, ox:ox + 512], selected_rgb):
            raise ValueError('Context RGB center differs from frozen tile')
        water = np.zeros((768, 768), dtype=np.float32)
        valid = np.zeros((768, 768), dtype=np.float32)
        water[oy:oy + 512, ox:ox + 512] = old_water == 255
        valid[oy:oy + 512, ox:ox + 512] = old_valid == 255
        image = (rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 - MEAN) / STD
        water, valid = (water[None], valid[None])
        if self.augment:
            image, water, valid = augment_triplet(image, water, valid)
        return (torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(water)), torch.from_numpy(np.ascontiguousarray(valid)))


# Original: src/external_train_768.py :: prepare_external_768
def _ext768_prepare_external_768(project: Path, external_rows: list[dict]) -> tuple[Path, str, dict[str, int]]:
    """Create or validate the fixed catalog before launching training."""
    project = Path(project).expanduser().resolve()
    _, frozen_rows, _ = _glh5406_load_screened_train_rows(project)
    if external_rows != frozen_rows:
        raise ValueError('External rows differ from frozen screened Train')
    destination = project / _ext768_DEFAULT_DEST
    if destination.exists():
        index = destination / 'index.csv'
        if not index.is_file():
            raise ValueError('Existing context destination is incomplete')
    else:
        index = _ext768_materialize_external_768_context(project, external_rows, destination)
    with (destination / 'summary.json').open(encoding='utf-8') as handle:
        summary = json.load(handle)
    if summary.get('rows') != len(external_rows) or summary.get('index_sha256') != _ext768__sha(index) or summary.get('supervision') != 'selected_512_only' or (summary.get('training_performed') is not False):
        raise ValueError('Context summary/index identity differs')
    reader = _ext768_ExternalContextDataset(project, external_rows, index, augment=False)
    counts = Counter(((item['source'], item['mode'], item['reason']) for item in reader.index))
    expected_counts = {f'{source}|{mode}|{reason}': count for (source, mode, reason), count in sorted(counts.items())}
    if summary.get('source_counts') != expected_counts:
        raise ValueError('Context summary counts differ')
    return (index, summary['index_sha256'], summary['source_counts'])



# Public API used by training entrypoints.
load_screened_train_rows = _glh5406_load_screened_train_rows
build_screened_train_dataset = _glh5406_build_screened_train_dataset
ScreenedMixedTrainDataset = _glh5406_ScreenedMixedTrainDataset
build_base_mixed_768_dataset = _base768_build_base_mixed_768_dataset
ExternalMixed768Dataset = _ext768_ExternalContextDataset
materialize_external_768_context = _ext768_materialize_external_768_context
prepare_external_768 = _ext768_prepare_external_768


def build_standard_training_dataset(project, base_rows, external_rows, data_config):
    """Original screened 5406 pairs, all provided as 512x512 training tiles."""
    return build_screened_train_dataset(project, base_rows, external_rows, data_config)



ROOT = Path(__file__).resolve().parents[2]


FIRST_RUN = 'glh_coast5406_from09_mixed768_40_20261102_001'


SECOND_RUN = 'glh_coast5406_from09_mixed768_best40_20261102_001'


SOURCE_SHA = 'f2eb899e7cd718f884f0add35f4948ae5f27b63bcdaf657f50342b949a246b7b'


TRAIN_SHA = '460833ec51f30d706c991b9b330f569f32d89f1a9266d42197da976e953c6e9f'


VAL_SHA = '5c285e82048e9733ee8e8428ad743a850beba7ac6ce6e9611b539018db56ce3c'


CONTEXT_INDEX = ROOT / 'data/processed/glh_coast_context768_20261102_001/index.csv'


EPOCHS = 40


STEPS = 5406


VAL_SCENES = 371


VAL_WINDOWS = 1337


THRESHOLD = 0.55


MIXED_POLICY = 'native_768_context_center_512_supervision_else_512'


CONTEXT_INDEX_SHA = 'dfb8bf32411f6fca8856d66e83aa198efcf74b7d55ef86e85b963e19d5812899'


def logical_epoch(stage: int, local_epoch: int) -> int:
    """Use 290..329, then 330..369 without repeating the first-stage order."""
    if type(stage) is not int or stage not in (1, 2) or type(local_epoch) is not int or (not 1 <= local_epoch <= EPOCHS):
        raise ValueError('stage must be 1/2 and local_epoch must be 1..40')
    return 289 + (stage - 1) * EPOCHS + local_epoch


def build_mixed_training_dataset(project: Path, base_rows: list[dict], external_rows: list[dict], data_config: dict, index_path: Path=CONTEXT_INDEX):
    """Preserve screened row order and substitute only real RGB context readers."""
    project = Path(project).expanduser().resolve()
    index_path = Path(index_path).expanduser().resolve()
    if data_config.get('train_manifest_sha256') != TRAIN_SHA or data_config.get('oem_val_sha256') != VAL_SHA or data_config.get('train_unique_pairs') != STEPS or (not index_path.is_file()) or (checksum(index_path) != CONTEXT_INDEX_SHA):
        raise ValueError('Mixed Train manifest/Validation/context index differs')
    old = build_screened_train_dataset(project, base_rows, external_rows, data_config)
    try:
        base = build_base_mixed_768_dataset(old.base)
        external = ExternalMixed768Dataset(project, external_rows, index_path)
        dataset = ScreenedMixedTrainDataset(base, external, old.rows)
        native = Counter(base.native768_counts)
        fallback = Counter(base.fallback512_counts)
        for row, context in zip(external_rows, external.index, strict=True):
            (native if context['mode'] == '768' else fallback)[row['source']] += 1
        if len(dataset) != STEPS or sum(native.values()) + sum(fallback.values()) != STEPS or native != Counter({'OEM': 1737, 'LandCover': 69, 'xBD': 54, 'GLH-Water': 310, 'CoastTrain_NAIP11': 674}) or (fallback != Counter({'OEM': 179, 'CAID': 1461, 'FLAIR': 523, 'GLH-Water': 378, 'CoastTrain_NAIP11': 21})):
            raise ValueError('Mixed Train context counts differ')
        return (dataset, {'native768_counts': dict(sorted(native.items())), 'fallback512_counts': dict(sorted(fallback.items()))})
    except Exception:
        old.close()
        raise


def build_run_config(stage: int, source: dict, data_config: dict, counts: dict) -> dict:
    """Pin source, unchanged factors and exact mixed-context membership."""
    logical_epoch(stage, 1)
    if data_config.get('train_manifest_sha256') != TRAIN_SHA or data_config.get('oem_val_sha256') != VAL_SHA or data_config.get('train_unique_pairs') != STEPS or (not isinstance(counts, dict)) or (sum(counts.get('native768_counts', {}).values()) != 2844) or (sum(counts.get('fallback512_counts', {}).values()) != 2562) or (source.get('source_checkpoint_sha256') is None) or (source.get('source_optimizer_step') is None):
        raise ValueError('Run source/data/context identity differs')
    run = FIRST_RUN if stage == 1 else SECOND_RUN
    source_path = Path(source['source_checkpoint'])
    if not source_path.is_relative_to(ROOT):
        raise ValueError('Source checkpoint is outside project')
    return {'run': run, 'purpose': 'epoch09_mixed_768_context_40_stage_' + str(stage), 'epochs': EPOCHS, 'start_epoch': 1, 'source_run': source['source_run'], 'source_epoch': source['source_epoch'], 'source_logical_epoch': 289 if stage == 1 else logical_epoch(1, source['source_epoch']), 'source_checkpoint': str(source_path.relative_to(ROOT)), 'source_checkpoint_sha256': source['source_checkpoint_sha256'], 'source_config_sha256': source['source_config_sha256'], 'source_history_sha256': source['source_history_sha256'], 'source_optimizer_step': source['source_optimizer_step'], 'optimizer_restored': True, 'optimizer_restarted': False, 'learning_rate': 5e-06, 'batch_size': 1, 'seed': 44, 'freeze_batchnorm': True, 'pos_weight': 1, 'negative_weight': 1.2, 'threshold': THRESHOLD, 'train_unique_pairs': STEPS, 'validation_scenes': VAL_SCENES, 'validation_windows': VAL_WINDOWS, 'train_manifest_sha256': TRAIN_SHA, 'external_file_index_sha256': data_config['external_file_index_sha256'], 'validation_csv_sha256': VAL_SHA, 'context_index': str(CONTEXT_INDEX.relative_to(ROOT)), 'context_index_sha256': CONTEXT_INDEX_SHA, 'mixed_train_policy': MIXED_POLICY, 'native768_counts': counts['native768_counts'], 'fallback512_counts': counts['fallback512_counts'], 'sampling_schedule_base_epoch': 289 if stage == 1 else 329, 'run_code_sha256': checksum(Path(__file__)), 'base_context_code_sha256': checksum(ROOT / 'src/training/mixed_data.py'), 'external_context_code_sha256': checksum(ROOT / 'src/training/mixed_data.py'), 'model_code_sha256': checksum(ROOT / 'src/model.py'), 'dataset_code_sha256': checksum(ROOT / 'src/dataset.py'), 'augmentation_code_sha256': checksum(ROOT / 'src/geometry_augmentation.py'), 'validation_code_sha256': checksum(ROOT / 'src/training/runner.py'), 'loss_code_sha256': checksum(ROOT / 'src/training/losses.py'), 'train_epoch_code_sha256': checksum(ROOT / 'src/training/runner.py'), 'keep_all_full': True, 'new_source_semantic_review_complete': False, 'glh_explicit_file_rights_confirmed': False}


def _check_training_tensors(dataset, model: torch.nn.Module, device: torch.device) -> list[str]:
    """Forward one row per source and backward once at 768 without stepping."""
    original = state_hash(model)
    seen = set()
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            source = row['source']
            if source in seen:
                continue
            image, water, valid = dataset[index]
            size = image.shape[-1]
            if size not in (512, 768) or tuple(image.shape) != (3, size, size) or tuple(water.shape) != (1, size, size) or (tuple(valid.shape) != (1, size, size)) or any((t.dtype != torch.float32 or not t.is_contiguous() or (not bool(torch.isfinite(t).all())) for t in (image, water, valid))) or (not bool(torch.all((water == 0) | (water == 1)))) or (not bool(torch.all((valid == 0) | (valid == 1)))) or (int(valid.sum()) == 0) or (size == 768 and int(valid.sum()) > 512 * 512):
                raise ValueError(f'Invalid mixed source tensor: {source}')
            logits = model(image[None].to(device))
            if tuple(logits.shape) != (1, 1, size, size) or not bool(torch.isfinite(logits).all()):
                raise ValueError(f'Invalid mixed source forward: {source}')
            seen.add(source)
            if len(seen) == 7:
                break
    if seen != {'OEM', 'CAID', 'LandCover', 'xBD', 'FLAIR', 'GLH-Water', 'CoastTrain_NAIP11'} or state_hash(model) != original:
        raise ValueError('Mixed source coverage/model state differs')
    first_native = next((i for i in range(len(dataset)) if i < 4023 and dataset.base.context_sizes[i] == 768))
    image, water, valid = dataset[first_native]
    configure_training_mode(model, True)
    logits = model(image[None].to(device))
    loss = negative_weighted_bce_12(logits, water[None].to(device))
    value = (loss * valid[None].to(device)).sum() / valid.sum().to(device)
    if not bool(torch.isfinite(value)):
        raise ValueError('Mixed 768 backward loss is nonfinite')
    value.backward()
    if not any((p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())):
        raise ValueError('Mixed 768 backward gradients are nonfinite')
    model.zero_grad(set_to_none=True)
    model.eval()
    if state_hash(model) != original:
        raise ValueError('Mixed preflight changed model state')
    return sorted(seen)
