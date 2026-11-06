from pathlib import Path
import sys
import rasterio
from rasterio.windows import Window


# 1. Получаем путь к изображению

if len(sys.argv) != 2:
    raise SystemExit(
        'Использование: python src/inspect_tiff.py /путь/к/изображению'
    )

image_path = Path(sys.argv[1]).expanduser()

if not image_path.is_file():
    raise FileNotFoundError(f"Файл не найден: {image_path}")


# 2. Создаём папку для результатов

output_dir = Path("outputs/checks") / image_path.stem
output_dir.mkdir(parents=True, exist_ok=True)


# 3. Открываем изображение

with rasterio.open(image_path) as src:

    print("\n=== ИНФОРМАЦИЯ ОБ ИЗОБРАЖЕНИИ ===")

    print("Формат:", src.driver)
    print("Размер:", src.width, "×", src.height)
    print("Количество каналов:", src.count)
    print("Тип данных:", src.dtypes)
    print("Цветовые каналы:", src.colorinterp)
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("NoData:", src.nodata)
    print("Сжатие:", src.compression)

    # 4. Выбираем размер фрагмента

    tile_size = min(512, src.width, src.height)

    positions = {
        "top_left": (0, 0),

        "top_right": (src.width - tile_size, 0),

        "center": (
            (src.width - tile_size) // 2,
            (src.height - tile_size) // 2
        ),

        "bottom_left": (0, src.height - tile_size),

        "bottom_right": (
            src.width - tile_size,
            src.height - tile_size
        )
    }

    # 5. Читаем и сохраняем фрагменты

    print("\n=== СОХРАНЯЕМ ФРАГМЕНТЫ ===")

    for name, (x, y) in positions.items():

        window = Window(
            col_off=x,
            row_off=y,
            width=tile_size,
            height=tile_size
        )

        # Читаем все каналы без преобразования в RGB
        tile = src.read(window=window)

        # Настройки выходного TIFF
        profile = {
            "driver": "GTiff",
            "width": tile_size,
            "height": tile_size,
            "count": src.count,
            "dtype": tile.dtype
        }

        # Если есть геопривязка, сохраняем её для фрагмента
        if src.crs is not None:
            profile["crs"] = src.crs
            profile["transform"] = src.window_transform(window)

        if src.nodata is not None:
            profile["nodata"] = src.nodata

        # Сохраняем фрагмент
        output_path = output_dir / f"{name}.tif"

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(tile)

        print(f"{name}: сохранён, X={x}, Y={y}")


print(f"\nГотово. Фрагменты сохранены в: {output_dir}")