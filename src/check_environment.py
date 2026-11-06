"""Проверка рабочего места для проекта water-segmentation.

Обычный запуск из папки проекта:
    python src/check_environment.py

Скрипт использует искусственные данные, а не спутниковый снимок.
Он проверяет загрузку ImageNet-весов, один шаг обучения U-Net на MPS,
получение предсказания и сохранение/чтение весов. Временные веса удаляются.
Это НЕ обучение распознаванию воды и НЕ оценка качества модели.

Дополнительный режим для отдельной проверки на CPU без скачивания весов:
    python src/check_environment.py --device cpu --random-weights
Этот режим НЕ подтверждает работу MPS или загрузку ImageNet.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    """Остановить проверку с понятным сообщением, если условие не выполнено."""
    if not condition:
        raise RuntimeError(message)


def run_checks(device_name: str, pretrained: bool, report: dict[str, Any]) -> None:
    # 1. Проверяем загрузку библиотек после установки PyTorch и SMP.
    print("[1/5] Проверяем Python и библиотеки...", flush=True)
    names = (
        "numpy", "rasterio", "geopandas", "pyogrio", "shapely",
        "matplotlib", "PIL", "torch", "torchvision", "segmentation_models_pytorch",
    )
    modules = {name: importlib.import_module(name) for name in names}
    report["versions"] = {name: str(mod.__version__) for name, mod in modules.items()}
    for name, version in report["versions"].items():
        print(f"  {name}: {version}", flush=True)
    print(f"  Python: {report['python']}")
    print(f"  Путь: {sys.executable}")
    print(f"  Архитектура: {platform.machine()}")

    torch = modules["torch"]
    np = modules["numpy"]
    smp = modules["segmentation_models_pytorch"]

    # 2. Не подменяем недоступный Apple GPU незаметным переходом на CPU.
    print(f"[2/5] Проверяем устройство {device_name}...", flush=True)
    if device_name == "mps":
        require(platform.system() == "Darwin", "Для проверки MPS нужен Mac.")
        require(torch.backends.mps.is_built(), "Эта сборка PyTorch не содержит MPS.")
        require(torch.backends.mps.is_available(), "PyTorch не может использовать MPS.")
        require(
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1",
            "Включён переход отдельных операций на CPU. Выполни в терминале "
            "unset PYTORCH_ENABLE_MPS_FALLBACK и запусти проверку заново.",
        )
    device = torch.device(device_name)

    # Проверяем совместную работу NumPy и PyTorch и передачу данных на устройство.
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    recovered = torch.from_numpy(original).to(device).cpu().numpy()
    np.testing.assert_array_equal(original, recovered)
    print(f"  Устройство: {device}; передача данных NumPy ↔ PyTorch: OK", flush=True)

    # 3. Создаём именно выбранную архитектуру. При первом запуске нужны интернет
    # и время для скачивания ImageNet-весов энкодера. Воды эти веса ещё не знают.
    print("[3/5] Создаём U-Net + ResNet-34...", flush=True)
    weights = "imagenet" if pretrained else None
    if pretrained:
        print("  Загружаем ImageNet-веса. При первом запуске возможна загрузка из сети.", flush=True)
    else:
        print("  ВНИМАНИЕ: используются случайные веса; ImageNet не проверяется.", flush=True)
    torch.manual_seed(42)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)
    report["model"] = {"architecture": "Unet", "encoder": "resnet34", "weights": weights}
    print("  Модель создана: OK", flush=True)

    # 4. Делаем один полный шаг обучения на одном искусственном изображении.
    # Маска с прямоугольником нужна только для проверки вычислений.
    print("[4/5] Выполняем пробный шаг обучения...", flush=True)
    size = 256
    images = torch.rand((1, 3, size, size), dtype=torch.float32, device=device)
    params = smp.encoders.get_preprocessing_params("resnet34", pretrained="imagenet")
    mean = torch.tensor(params["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(params["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    target = torch.zeros((1, 1, size, size), dtype=torch.float32, device=device)
    target[:, :, size // 4:3 * size // 4, size // 4:3 * size // 4] = 1

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tracked_name, tracked_weight = next(model.named_parameters())
    before = tracked_weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    require(tuple(logits.shape) == tuple(target.shape), "Не совпадают размеры предсказания и маски.")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    require(bool(torch.isfinite(loss).item()), "Вместо значения ошибки получено NaN или Inf.")
    loss.backward()
    require(tracked_weight.grad is not None, "Ошибка не дошла до параметров энкодера.")
    require(bool(torch.isfinite(tracked_weight.grad).all().item()), "В градиентах появились NaN или Inf.")
    require(bool((tracked_weight.grad != 0).any().item()), "Все проверяемые градиенты равны нулю.")
    optimizer.step()
    if device_name == "mps":
        torch.mps.synchronize()
    change = float((tracked_weight.detach() - before).abs().max().item())
    require(change > 0, "Проверяемые веса не изменились после шага обучения.")
    report["training_check"] = {
        "input_shape": list(images.shape), "output_shape": list(logits.shape),
        "loss": float(loss.item()), "checked_parameter": tracked_name,
        "max_weight_change": change,
    }
    print(f"  Выход: {tuple(logits.shape)}; веса изменились: OK", flush=True)
    del optimizer, logits, loss, before, tracked_weight

    # 5. Проверяем режим предсказания и сохранение только значений весов.
    # Создаём временный файл, загружаем его в новую модель и сравниваем результаты.
    # Временный файл удалится автоматически: использовать его для проекта нельзя.
    print("[5/5] Проверяем предсказание и сохранение/чтение весов...", flush=True)
    model.eval()
    with torch.inference_mode():
        expected = model(images).sigmoid().cpu()
    require(bool(torch.isfinite(expected).all().item()), "В предсказании появились NaN или Inf.")
    require(bool(((expected >= 0) & (expected <= 1)).all().item()), "Оценки вне диапазона 0–1.")

    with tempfile.TemporaryDirectory(prefix="water-seg-check-") as tmp:
        checkpoint = Path(tmp) / "temporary_test_weights.pt"
        cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        torch.save(cpu_state, checkpoint)
        restored = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=1, activation=None,
        )
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        restored.to(device).eval()
        with torch.inference_mode():
            actual = restored(images).sigmoid().cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    report["save_load_check"] = "passed"
    print("  Предсказание и восстановление модели: OK", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--random-weights", action="store_true")
    args = parser.parse_args()
    # Файл должен лежать в src/ внутри папки проекта.
    root = Path(__file__).resolve().parent.parent
    output = root / "outputs" / "checks"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "environment_check.json"
    report: dict[str, Any] = {
        "status": "running",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "system": platform.platform(),
        "architecture": platform.machine(),
        "device": args.device,
        "pretrained_requested": not args.random_weights,
        "note": "Synthetic environment check only. Not water-training results.",
    }
    code = 0
    try:
        run_checks(args.device, not args.random_weights, report)
        report["status"] = "passed"
        print("\nПРОВЕРКА СРЕДЫ: OK", flush=True)
        print("Это проверка работы кода, а не качества распознавания воды.", flush=True)
    except Exception as exc:
        code = 1
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print("\nПРОВЕРКА ОСТАНОВЛЕНА. Ниже полный текст ошибки.", flush=True)
        traceback.print_exc(file=sys.stdout)
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Отчёт: {report_path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
