from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.windows import Window

import torch
from torch.utils.data import Dataset
from geometry_augmentation import augment_triplet


# Настройки нашего датасета

TILE_SIZE = 512
WATER_CLASS = 6

# Нормализация RGB для ResNet-34 с ImageNet-весами.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class WaterDataset(Dataset):

    def __init__(self, csv_path, data_dir, mode="train"):

        if mode not in ("train", "val", "test"):
            raise ValueError("mode должен быть train, val или test")

        self.data_dir = Path(data_dir)
        self.mode = mode
        self.tile_size = TILE_SIZE

        # Читаем список изображений.
        with open(csv_path, newline="", encoding="utf-8") as file:
            self.samples = list(csv.DictReader(file))

        if not self.samples:
            raise ValueError("Список изображений пуст")

        # Для Validation и Test заранее определяем все окна.
        # Так мы проверяем изображение целиком, а не случайный кусок.
        self.windows = []

        if mode != "train":

            for index, sample in enumerate(self.samples):

                image_path = self.data_dir / sample["image"]

                with rasterio.open(image_path) as src:
                    width = src.width
                    height = src.height

                for y in range(0, height, TILE_SIZE):
                    for x in range(0, width, TILE_SIZE):
                        self.windows.append((index, x, y))

    def __len__(self):

        if self.mode == "train":
            return len(self.samples)

        return len(self.windows)

    def __getitem__(self, index):

        # 1. Выбираем изображение.
        if self.mode == "train":
            sample = self.samples[index]
        else:
            sample_index, x, y = self.windows[index]
            sample = self.samples[sample_index]

        image_path = self.data_dir / sample["image"]
        mask_path = self.data_dir / sample["mask"]

        # 2. Открываем изображение и маску.
        with rasterio.open(image_path) as image_src, \
             rasterio.open(mask_path) as mask_src:

            width = image_src.width
            height = image_src.height

            if image_src.count != 3:
                raise ValueError(f"Ожидалось RGB: {image_path}")

            if mask_src.count != 1:
                raise ValueError(f"Ожидалась одноканальная маска: {mask_path}")

            if mask_src.dtypes != ("uint8",):
                raise ValueError(f"Ожидалась маска uint8: {mask_path}")

            if (width, height) != (mask_src.width, mask_src.height):
                raise ValueError("Размеры изображения и маски отличаются")

            if image_src.crs is not None and mask_src.crs is not None:
                if (image_src.crs != mask_src.crs
                        or not image_src.transform.almost_equals(mask_src.transform)):
                    raise ValueError("Геопривязка изображения и маски различается")

            # 3. Выбираем фрагмент для обучения.
            if self.mode == "train":

                # В некоторых случаях специально выбираем воду,
                # чтобы модель не училась преимущественно на суше.
                x, y = self._random_position(mask_src)

            # 4. Создаём окно чтения.
            window = Window(
                x,
                y,
                min(TILE_SIZE, width - x),
                min(TILE_SIZE, height - y)
            )

            # Читаем исходные значения пикселей.
            image = image_src.read([1, 2, 3], window=window)
            mask = mask_src.read(1, window=window)

            # Получаем информацию о недействительных пикселях.
            image_valid = np.all(
                image_src.read_masks([1, 2, 3], window=window) > 0,
                axis=0,
            )
            mask_valid = mask_src.read_masks(1, window=window) > 0

        if image.dtype != np.uint8:
            raise ValueError(f"Ожидалось RGB uint8: {image_path}")

        if not np.all((mask >= 0) & (mask <= 8)):
            raise ValueError(f"Неизвестные номера классов: {mask_path}")

        # По умолчанию RGB не меняется; отдельный обучающий subclass может
        # изменить его до нормализации, не затрагивая Validation и Test.
        if self.mode == "train":
            image = self._augment_rgb(image)

        # 5. Создаём бинарную маску воды.
        water = (mask == WATER_CLASS).astype(np.float32)

        # Известные классы: 1–8.
        # Класс 0 считается неизвестным.
        valid = (
            (mask >= 1)
            & (mask <= 8)
            & image_valid
            & mask_valid
        ).astype(np.float32)

        # 6. Подготавливаем RGB для предобученного энкодера.
        image = image.astype(np.float32)
        image /= 255.0
        image -= MEAN
        image /= STD

        # 7. Дополняем маленькие фрагменты до 512×512.
        h, w = water.shape

        if h == TILE_SIZE and w == TILE_SIZE:
            image_out = image
            water_out = water.reshape(1, h, w)
            valid_out = valid.reshape(1, h, w)
        else:
            image_out = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
            water_out = np.zeros((1, TILE_SIZE, TILE_SIZE), dtype=np.float32)
            valid_out = np.zeros((1, TILE_SIZE, TILE_SIZE), dtype=np.float32)

            image_out[:, :h, :w] = image
            water_out[0, :h, :w] = water
            valid_out[0, :h, :w] = valid

        # 8. Разнообразим изображения только во время обучения.

        if self.mode == "train":

            image_out, water_out, valid_out = augment_triplet(
                image_out, water_out, valid_out)

        # Создаём массивы с подходящим расположением данных в памяти.
        image_out = np.ascontiguousarray(image_out)
        water_out = np.ascontiguousarray(water_out)
        valid_out = np.ascontiguousarray(valid_out)
       

        # 8. Превращаем массивы в данные PyTorch.
        return (
            torch.from_numpy(image_out),
            torch.from_numpy(water_out),
            torch.from_numpy(valid_out)
        )

    def _augment_rgb(self, rgb):
        return rgb

    def _random_position(self, mask_src):

        width = mask_src.width
        height = mask_src.height

        max_x = max(0, width - TILE_SIZE)
        max_y = max(0, height - TILE_SIZE)

        # С вероятностью 50% пытаемся выбрать фрагмент,
        # содержащий воду.
        if np.random.random() < 0.5:

            labels = mask_src.read(1)
            flat = np.flatnonzero(labels == WATER_CLASS)

            if flat.size > 0:

                index = np.random.randint(flat.size)

                py, px = divmod(int(flat[index]), width)

                # Выбираем начало окна так,
                # чтобы выбранный пиксель воды оказался внутри.
                x_min = max(0, px - TILE_SIZE + 1)
                x_max = min(px, max_x)

                y_min = max(0, py - TILE_SIZE + 1)
                y_max = min(py, max_y)

                x = np.random.randint(x_min, x_max + 1)
                y = np.random.randint(y_min, y_max + 1)

                return x, y

        # В остальных случаях выбираем случайное окно.
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)

        return x, y
