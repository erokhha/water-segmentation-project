from pathlib import Path
import sys
import rasterio
from rasterio.windows import Window


if len(sys.argv) != 2:
    raise SystemExit(
        'Использование: python src/inspect_tiff.py /путь/к/изображению'
    )

image_path = Path(sys.argv[1]).expanduser()

if not image_path.is_file():
    raise FileNotFoundError(f"Файл не найден: {image_path}")



output_dir = Path("outputs/checks") / image_path.stem
output_dir.mkdir(parents=True, exist_ok=True)



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


    print("\n=== СОХРАНЯЕМ ФРАГМЕНТЫ ===")

    for name, (x, y) in positions.items():

        window = Window(
            col_off=x,
            row_off=y,
            width=tile_size,
            height=tile_size
        )

        tile = src.read(window=window)


        profile = {
            "driver": "GTiff",
            "width": tile_size,
            "height": tile_size,
            "count": src.count,
            "dtype": tile.dtype
        }

        if src.crs is not None:
            profile["crs"] = src.crs
            profile["transform"] = src.window_transform(window)

        if src.nodata is not None:
            profile["nodata"] = src.nodata

        output_path = output_dir / f"{name}.tif"

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(tile)

        print(f"{name}: сохранён, X={x}, Y={y}")


print(f"\nГотово. Фрагменты сохранены в: {output_dir}")