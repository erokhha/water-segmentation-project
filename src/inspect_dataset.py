"""Проверка OpenEarthMap 

Запуск из проекта:
    python src/inspect_dataset.py
    python src/inspect_dataset.py --data data/raw/openearthmap

Коды классов проверены по авторскому open_earth_map/utils.py:
https://github.com/bao18/open_earth_map/blob/main/open_earth_map/utils.py
Структура пар проверена по авторскому open_earth_map/dataset.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning

PROJECT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    0: "unknown", 1: "bareland", 2: "rangeland", 3: "developed_space",
    4: "road", 5: "tree", 6: "water", 7: "agriculture_land", 8: "building",
}
WATER_CODE = 6
PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148],
    [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7],
], dtype=np.uint8)


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_pair(image_path: Path, mask_path: Path):
    with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
        if image_src.count != 3 or image_src.dtypes != ("uint8",) * 3:
            raise ValueError("Ожидалось RGB из 3 каналов uint8; автоматического преобразования нет")
        if mask_src.count != 1 or mask_src.dtypes != ("uint8",):
            raise ValueError("Ожидалась одноканальная маска uint8 с номерами классов")
        if (image_src.width, image_src.height) != (mask_src.width, mask_src.height):
            raise ValueError("Размеры изображения и маски не совпадают")
        if image_src.crs is not None and mask_src.crs is not None:
            if image_src.crs != mask_src.crs or not image_src.transform.almost_equals(mask_src.transform):
                raise ValueError("Геопривязка изображения и маски различается")
        rgb = image_src.read()
        labels = mask_src.read(1)
        codes = np.unique(labels)
        if np.any(codes > 8):
            raise ValueError(f"Неожиданные коды разметки: {codes.tolist()}; ожидались 0..8")
        valid = (labels != 0) & (mask_src.read_masks(1) > 0)
        valid &= np.all(image_src.read_masks() > 0, axis=0)
        colors = ",".join(color.name for color in image_src.colorinterp)
    return rgb, labels, valid, colors


def source_splits(root: Path) -> dict[str, set[str]]:
    result = defaultdict(set)
    for filename, name in (("train.txt", "train"), ("val.txt", "val"), ("test.txt", "test")):
        for path in root.rglob(filename):
            if "__MACOSX" in path.parts:
                continue
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    result[Path(line).stem].add(name)
    return result


def choose_examples(rows: list[dict], limit: int) -> list[dict]:
    """Для осмотра, не для обучения: разные регионы с водой плюс немного суши."""
    good = [row for row in rows if row["status"] == "read_ok"]

    unique = {}
    for row in good:
        unique.setdefault(row["image_sha256"], row)
    good = list(unique.values())
    with_water = sorted(
        [r for r in good if r["water_pixels"] > 0],
        key=lambda r: (abs(r["water_fraction"] - 0.4), r["sample_id"]),
    )
    land = sorted(
        [r for r in good if r["water_pixels"] == 0],
        key=lambda r: (-r["building_pixels"], r["sample_id"]),
    )
    chosen, used_regions, used_ids = [], set(), set()
    water_limit = max(1, limit - min(4, len(land)))

    for row in with_water:
        if row["region"] not in used_regions and len(chosen) < water_limit:
            chosen.append(row)
            used_regions.add(row["region"])
            used_ids.add(row["sample_id"])

    for row in with_water:
        if len(chosen) >= water_limit:
            break
        if row["sample_id"] not in used_ids:
            chosen.append(row)
            used_ids.add(row["sample_id"])
    for row in land:
        if len(chosen) >= limit:
            break
        chosen.append(row)
    return chosen


def make_gallery(rows: list[dict], root: Path, output: Path, limit: int) -> int:
    selected = choose_examples(rows, limit)
    preview_dir = output / "previews"
    preview_dir.mkdir(exist_ok=True)
    cards, review_rows = [], []
    for number, row in enumerate(selected, 1):
        rgb, labels, valid, _ = load_pair(root / row["image"], root / row["mask"])
        rgb = np.moveaxis(rgb, 0, -1)

        water_preview = np.full(labels.shape, 128, dtype=np.uint8)
        water_preview[valid] = 0
        water_preview[valid & (labels == WATER_CODE)] = 255
        images = {"rgb": rgb, "classes": PALETTE[labels], "water": water_preview}
        descriptions = {"rgb": "Исходное RGB", "classes": "Классы авторов", "water": "Вода / не вода / неизвестно"}
        figures = []
        for kind, array in images.items():
            name = f"{number:02d}_{kind}.png"
            Image.fromarray(array).save(preview_dir / name)
            link = f"previews/{name}"
            figures.append(
                f'<figure><figcaption>{descriptions[kind]}</figcaption>'
                f'<a href="{link}" target="_blank"><img src="{link}" loading="lazy"></a></figure>'
            )
        sid = html.escape(row["sample_id"])
        cards.append(
            f'<section><h2>{number:02d}. {sid}</h2>'
            f'<p>{row["width"]} × {row["height"]} px; '
            f'вода: {100 * row["water_fraction"]:.2f}% допустимых пикселей. '
            'Для мелких объектов откройте PNG по нажатию и смотрите в масштабе 100%.</p>'
            '<div class="images">' + "".join(figures) + '</div></section>'
        )
        review_rows.append({
            "example": f"{number:02d}", "sample_id": row["sample_id"],
            "region": row["region"], "alignment_ok": "", "water_label_ok": "",
            "boats_visible": "", "boats_excluded": "", "piers_visible": "",
            "piers_excluded": "", "islands_visible": "", "islands_excluded": "", "notes": "",
        })
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>OpenEarthMap: проверка разметки</title><style>
body{font:16px system-ui,sans-serif;margin:24px;line-height:1.45}
.images{display:flex;gap:14px;flex-wrap:wrap} figure{margin:0;width:32%;min-width:260px}
img{width:100%;height:auto;image-rendering:pixelated}section{margin:36px 0}figcaption{font-weight:bold}
</style><h1>OpenEarthMap: проверка разметки, не результат нейросети</h1>
<p>Исходные файлы не изменены. Чёрно-белая картинка: белое = water (6),
чёрное = другие известные классы, серое = неизвестные/отсутствующие данные.</p>
<p>Эти примеры выбраны для осмотра. Они не гарантируют присутствия лодок,
приемлемого качества всего набора или репрезентативности будущего теста.</p>
''' + "\n".join(cards) + "</html>"
    (output / "index.html").write_text(document, encoding="utf-8")
    fields = ["example", "sample_id", "region", "alignment_ok", "water_label_ok",
              "boats_visible", "boats_excluded", "piers_visible", "piers_excluded",
              "islands_visible", "islands_excluded", "notes"]

    save_csv(output / "review_template.csv", review_rows, fields)
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/raw/openearthmap")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs/checks/openearthmap")
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()
    root, output = args.data.expanduser().resolve(), args.out.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Нет папки данных: {root}. Сначала распакуйте архив.")
    if root == output or root in output.parents:
        raise SystemExit("Отчёты должны лежать вне папки исходного датасета.")
    if not 1 <= args.examples <= 60:
        raise SystemExit("--examples должен быть между 1 и 60")
    output.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    regions = sorted({
        p.parent for p in root.rglob("*")
        if p.is_dir() and p.name in {"images", "labels"} and "__MACOSX" not in p.parts
    })
    if not regions:
        raise SystemExit("Не найдены папки images/labels. Пришлите структуру распакованного архива.")
    original_splits = source_splits(root)
    rows, issues, seen_ids = [], [], set()
    total_codes = Counter()
    print("[1/3] Читаем пары изображений и масок. Один файл за раз.", flush=True)
    for region_dir in regions:
        files = {}
        for kind in ("images", "labels"):
            folder = region_dir / kind
            group = {}
            if folder.is_dir():
                for path in sorted(folder.iterdir()):
                    if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in {".tif", ".tiff"}:
                        if path.stem in group:
                            raise ValueError(f"Повтор имени изображения: {path.stem} в {folder}")
                        group[path.stem] = path
            files[kind] = group
        for stem in sorted(set(files["images"]) | set(files["labels"])):
            sid = f"{region_dir.name}/{stem}"
            if sid in seen_ids:
                raise ValueError(f"Найдено несколько копий {sid}. Укажите одну распакованную версию через --data.")
            seen_ids.add(sid)
            image, mask = files["images"].get(stem), files["labels"].get(stem)
            row = {
                "sample_id": sid, "region": region_dir.name,
                "image": image.relative_to(root).as_posix() if image else "",
                "mask": mask.relative_to(root).as_posix() if mask else "",
                "source_split": ",".join(sorted(original_splits.get(stem, set()))),
            }
            if image is None or mask is None:
                row["status"] = "missing_rgb" if image is None else "missing_label"
                issues.append({"sample_id": sid, "kind": row["status"],
                               "detail": "Пары нет; причину нужно сверить с составом официального набора"})
            else:
                try:
                    rgb, labels, valid, colors = load_pair(image, mask)
                    valid_count = int(valid.sum())
                    water_count = int(((labels == WATER_CODE) & valid).sum())
                    digest = hashlib.sha256()
                    digest.update(str(rgb.shape).encode("ascii"))
                    digest.update(rgb.tobytes())
                    row.update({
                        "status": "read_ok" if valid_count else "no_valid_pixels",
                        "width": labels.shape[1], "height": labels.shape[0],
                        "colorinterp": colors, "codes": ",".join(map(str, np.unique(labels).tolist())),
                        "valid_pixels": valid_count, "water_pixels": water_count,
                        "building_pixels": int(((labels == 8) & valid).sum()),
                        "water_fraction": water_count / valid_count if valid_count else 0,
                        "ignored_fraction": 1 - valid_count / labels.size,
                        "image_sha256": digest.hexdigest(),
                        "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                    })
                    total_codes.update({int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})
                    if not valid_count:
                        issues.append({"sample_id": sid, "kind": "no_valid_pixels", "detail": "Все пиксели неизвестны или объявлены отсутствующими"})
                    if colors != "red,green,blue":
                        issues.append({"sample_id": sid, "kind": "colorinterp_check", "detail": colors})
                except (ValueError, OSError, rasterio.errors.RasterioError) as error:
                    row["status"] = "read_error"
                    issues.append({"sample_id": sid, "kind": "read_error", "detail": str(error).replace(str(root), "<dataset>")})
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f"  Проверено записей: {len(rows)}", flush=True)
    if not rows:
        raise SystemExit("В папках images/labels не найдено TIFF-файлов. Пришлите структуру архива.")
    status_counts = Counter(r["status"] for r in rows)
    by_hash = defaultdict(list)
    region_summary = defaultdict(Counter)
    for row in rows:
        counter = region_summary[row["region"]]
        counter[row["status"]] += 1
        if row["status"] == "read_ok":
            by_hash[row["image_sha256"]].append(row)
            for key in ("valid_pixels", "water_pixels"):
                counter[key] += row[key]
    duplicates = []
    for digest, group in by_hash.items():
        if len(group) > 1:
            duplicates.append({"image_sha256": digest, "sample_ids": [r["sample_id"] for r in group],
                               "same_labels": len({r["label_sha256"] for r in group}) == 1})
    inventory_fields = ["sample_id", "region", "image", "mask", "source_split", "status",
                        "width", "height", "colorinterp", "codes", "valid_pixels", "water_pixels",
                        "building_pixels", "water_fraction", "ignored_fraction", "image_sha256", "label_sha256"]
    save_csv(output / "inventory.csv", rows, inventory_fields)
    save_csv(output / "issues.csv", issues, ["sample_id", "kind", "detail"])
    save_csv(output / "regions.csv", [{"region": k, **dict(v)} for k, v in sorted(region_summary.items())],
             ["region", "read_ok", "missing_rgb", "missing_label", "read_error", "no_valid_pixels", "valid_pixels", "water_pixels"])
    print("[2/3] Готовим примеры в исходном размере для визуальной проверки.", flush=True)
    example_count = make_gallery(rows, root, output, args.examples)
    print("[3/3] Сохраняем отчёт.", flush=True)
    good_rows = [r for r in rows if r["status"] == "read_ok"]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "OpenEarthMap v1",
        "class_codes": CLASS_NAMES, "water_code": WATER_CODE, "unknown_code": 0,
        "records": len(rows), "status_counts": dict(status_counts),
        "regions_with_pairs": len({r["region"] for r in good_rows}),
        "pairs_with_water": sum(r["water_pixels"] > 0 for r in good_rows),
        "pixels_by_original_code_in_readable_pairs": dict(sorted(total_codes.items())),
        "duplicate_rgb_groups": duplicates, "preview_examples": example_count,
        "technical_read_errors": status_counts["read_error"],
        "versions": {"numpy": np.__version__, "rasterio": rasterio.__version__},
        "not_checked": ["Ручное качество масок и исключение лодок/причалов", "Лицензии выбранных источников",
                        "Географические пересечения похожих, но не идентичных снимков", "Наше разделение Train/Validation/Test"],
        "status": "inspection_only_not_training_approved",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(rows), "status_counts": dict(status_counts),
                      "pairs_with_water": summary["pairs_with_water"], "duplicate_rgb_groups": len(duplicates),
                      "preview_examples": example_count}, ensure_ascii=False, indent=2))
    print(f"\nОтчёт: {output / 'summary.json'}\nПросмотр: {output / 'index.html'}")
    print("Это проверка данных, не разрешение запускать обучение.")
    return 2 if status_counts["read_error"] or not good_rows else 0


if __name__ == "__main__":
    sys.exit(main())
