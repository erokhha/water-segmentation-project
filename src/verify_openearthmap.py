"""Проверить скачанный официальный архив OpenEarthMap v1 без распаковки.

Источник контрольной суммы:
https://zenodo.org/records/7223446

Запуск: python src/verify_openearthmap.py
Или:    python src/verify_openearthmap.py --archive /путь/OpenEarthMap.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_MD5 = "64155d1dc9d3b68536063f79878e1a67"


def file_md5(path: Path) -> str:
    # MD5 используется для сравнения с опубликованным архивом, не для паролей.
    digest = hashlib.md5(usedforsecurity=False)
    size = path.stat().st_size
    read = 0
    next_notice = 1024 ** 3
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 ** 2):
            digest.update(chunk)
            read += len(chunk)
            if read >= next_notice:
                print(f"Проверено {read / 1024**3:.1f} из {size / 1024**3:.1f} GiB", flush=True)
                next_notice += 1024 ** 3
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=PROJECT / "data/raw/OpenEarthMap.zip")
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"Файл не найден: {archive}")
    print("Сравниваем контрольную сумму с официальной версией v1.", flush=True)
    actual = file_md5(archive)
    print("MD5:", actual)
    if actual != EXPECTED_MD5:
        raise SystemExit("ОШИБКА: контрольная сумма не совпала. Не распаковывайте архив и не запускайте следующие шаги.")
    # Центральный каталог ZIP позволяет узнать размер после распаковки.
    # Целостность всего файла уже проверена по совпавшей MD5.
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        file_count = sum(not m.is_dir() for m in members)
        unpacked_bytes = sum(m.file_size for m in members)
    report = {
        "dataset": "OpenEarthMap v1", "source": "https://zenodo.org/records/7223446",
        "checked_utc": datetime.now(timezone.utc).isoformat(), "archive": archive.name,
        "expected_md5": EXPECTED_MD5, "actual_md5": actual, "md5_matches": True,
        "archive_bytes": archive.stat().st_size,
        "uncompressed_bytes_in_zip": unpacked_bytes, "files_in_zip": file_count,
        "note": "Архив проверен. Распаковка, качество масок и пригодность для обучения этой командой не проверяются.",
    }
    output = PROJECT / "outputs/checks/openearthmap_archive.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MD5: OK")
    print(f"Размер файлов после распаковки: {unpacked_bytes / 1024**3:.2f} GiB")
    print("Отчёт:", output)


if __name__ == "__main__":
    main()
