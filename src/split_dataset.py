"""Разделение OpenEarthMap по регионам.

Создаёт списки Train, Validation и Test.
Не изменяет и не копирует исходные изображения.
"""

from pathlib import Path
from collections import defaultdict
import csv
import json
import random


# 1. Пути к данным и результатам

PROJECT = Path(__file__).resolve().parents[1]

INVENTORY = PROJECT / "outputs/checks/openearthmap/inventory.csv"
DATA = PROJECT / "data/raw/openearthmap"
OUT = PROJECT / "configs/splits"

SEED = 42


def main(argv=None):
    # 2. Читаем результаты предыдущей проверки

    if not INVENTORY.is_file():
        raise SystemExit(
            f"Не найден отчёт предыдущего этапа: {INVENTORY}"
        )

    with INVENTORY.open(encoding="utf-8-sig", newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["status"] == "read_ok"
        ]

    if not rows:
        raise SystemExit(
            "В inventory.csv не найдено ни одной полной пары."
        )


    # 3. Получаем список регионов

    regions = sorted({row["region"] for row in rows})

    # Перемешиваем регионы воспроизводимым способом
    random.Random(SEED).shuffle(regions)

    # Примерно 70% регионов для обучения,
    # 15% для настройки, 15% для проверки.
    n_val = round(len(regions) * 0.15)
    n_test = round(len(regions) * 0.15)

    sets = {
        "test": set(regions[:n_test]),
        "val": set(regions[n_test:n_test + n_val]),
        "train": set(regions[n_test + n_val:]),
    }


    # 4. Проверяем, что регионы не пересекаются

    assert len(set.union(*sets.values())) == len(regions)

    assert all(
        not (sets[a] & sets[b])
        for a in sets
        for b in sets
        if a != b
    )


    # 5. Распределяем изображения по группам

    by_split = defaultdict(list)

    seen_ids = set()
    seen_hash = {}

    for row in rows:

        sid = row["sample_id"]

        if sid in seen_ids:
            raise SystemExit(
                f"Повторяющийся идентификатор: {sid}"
            )

        seen_ids.add(sid)

        split = next(
            name
            for name, group in sets.items()
            if row["region"] in group
        )

        # Проверяем существование исходных файлов
        for key in ("image", "mask"):

            path = DATA / row[key]

            if not path.is_file():
                raise SystemExit(
                    f"Не найден {key}: {path}"
                )

        # Проверяем точные дубликаты изображений
        image_hash = row["image_sha256"]

        if (
            image_hash in seen_hash
            and seen_hash[image_hash] != split
        ):
            raise SystemExit(
                f"Точные копии изображения в разных выборках: {sid}"
            )

        seen_hash[image_hash] = split

        by_split[split].append(row)


    # 6. Создаём папку результатов

    OUT.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample_id",
        "region",
        "image",
        "mask",
        "source_split",
        "valid_pixels",
        "water_pixels",
        "water_fraction",
        "image_sha256",
        "label_sha256",
    ]

    summary = {
        "seed": SEED,
        "source": "OpenEarthMap v1",
        "split_by": "region",
        "training_approved": False,
    }


    # 7. Сохраняем три списка

    for name in ("train", "val", "test"):

        group_rows = sorted(
            by_split[name],
            key=lambda row: row["sample_id"]
        )

        # В каждой выборке должна присутствовать вода
        if (
            not group_rows
            or not any(
                int(row["water_pixels"]) > 0
                for row in group_rows
            )
        ):
            raise SystemExit(
                f"Выборка {name} пуста или в ней нет воды."
            )

        output_path = OUT / f"{name}.csv"

        with output_path.open(
            "w", encoding="utf-8", newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
                extrasaction="ignore"
            )

            writer.writeheader()
            writer.writerows(group_rows)

        # Считаем статистику
        n_valid = sum(
            int(row["valid_pixels"])
            for row in group_rows
        )

        n_water = sum(
            int(row["water_pixels"])
            for row in group_rows
        )

        summary[name] = {
            "regions": sorted(sets[name]),
            "images": len(group_rows),
            "images_with_water": sum(
                int(row["water_pixels"]) > 0
                for row in group_rows
            ),
            "water_fraction": round(n_water / n_valid, 5),
        }

        print(
            f"{name:5} | "
            f"регионов: {len(sets[name]):2} | "
            f"пар: {len(group_rows):4} | "
            f"с водой: {summary[name]['images_with_water']:4} | "
            f"доля воды: {100 * n_water / n_valid:.2f}%"
        )


    # 8. Сохраняем общую информацию о разделении

    summary_path = OUT / "summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print("\nРезультат:", OUT)
    print("Исходные изображения не изменены.")
    print("Перед обучением нужно проверить лицензии регионов.")

if __name__ == "__main__":
    main()
