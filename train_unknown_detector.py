import argparse
import json
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# ============================================================
# НАСТРОЙКИ
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
CUSTOM_DATA_DIR = SCRIPT_DIR / "detector_dataset"
CUSTOM_DIGITS_DIR = CUSTOM_DATA_DIR / "digits"
CUSTOM_UNKNOWN_DIR = CUSTOM_DATA_DIR / "unknown"

MODEL_FILE = SCRIPT_DIR / "unknown_detector.pth"
CONFIG_FILE = SCRIPT_DIR / "unknown_detector_config.json"
REPORT_FILE = SCRIPT_DIR / "unknown_detector_report.txt"

DIGIT_THRESHOLD = 0.70
EPOCHS = 5
LEARNING_RATE = 0.001
SYNTHETIC_TRAIN_COUNT = 50_000
SYNTHETIC_VALIDATION_COUNT = 10_000
CUSTOM_TRAIN_REPEATS = 50
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


# ============================================================
# АРХИТЕКТУРА: СОВПАДАЕТ С digit_draw_4.py
# ============================================================

class UnknownDetector(nn.Module):
    """Вероятность того, что изображение является цифрой."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


# ============================================================
# ОБРАБОТКА 28x28 — ТАКАЯ ЖЕ, КАК В ОСНОВНОЙ ПРОГРАММЕ
# ============================================================

def center_by_mass(image):
    pixels = image.load()
    total = 0
    sum_x = 0
    sum_y = 0

    for y in range(28):
        for x in range(28):
            value = pixels[x, y]
            total += value
            sum_x += x * value
            sum_y += y * value

    if total == 0:
        return image

    shift_x = round(13.5 - sum_x / total)
    shift_y = round(13.5 - sum_y / total)
    centered = Image.new("L", (28, 28), 0)
    centered.paste(image, (shift_x, shift_y))
    return centered


def prepare_image(image):
    image = image.convert("L")

    # Если фон светлый, делаем его чёрным, а штрихи белыми.
    pixels = list(image.getdata())
    if pixels and sum(pixels) / len(pixels) > 127:
        image = Image.eval(image, lambda value: 255 - value)

    bbox = image.getbbox()
    if bbox is None:
        return None

    image = image.crop(bbox)
    width, height = image.size
    scale = 20 / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    result = Image.new("L", (28, 28), 0)
    result.paste(image, ((28 - new_width) // 2, (28 - new_height) // 2))
    return center_by_mass(result)


# ============================================================
# СБОР СОБСТВЕННЫХ ПРИМЕРОВ
# ============================================================

def run_collector():
    import tkinter as tk
    from tkinter import messagebox

    CUSTOM_DIGITS_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)

    size = 420
    brush_size = 12
    drawing_image = Image.new("L", (size, size), 0)
    drawing = ImageDraw.Draw(drawing_image)
    last_position = [None, None]

    root = tk.Tk()
    root.title("Сбор примеров для детектора")

    title = tk.Label(
        root,
        text="Нарисуйте ОДИН символ",
        font=("Arial", 18, "bold"),
    )
    title.pack(pady=(12, 4))

    instruction = tk.Label(
        root,
        text=(
            "Для букв, знаков и каракулей нажмите «Сохранить: НЕИЗВЕСТНЫЙ».\n"
            "Для цифр 0–9 нажмите «Сохранить: ЦИФРА»."
        ),
        font=("Arial", 11),
        justify=tk.CENTER,
    )
    instruction.pack(pady=(0, 8))

    canvas = tk.Canvas(root, width=size, height=size, bg="black")
    canvas.pack(padx=15)

    counter_label = tk.Label(root, font=("Arial", 11))
    counter_label.pack(pady=6)

    def update_counter():
        digit_count = len(list(CUSTOM_DIGITS_DIR.glob("*.png")))
        unknown_count = len(list(CUSTOM_UNKNOWN_DIR.glob("*.png")))
        counter_label.config(
            text=f"Сохранено: цифр {digit_count}; неизвестных {unknown_count}"
        )

    def start_draw(event):
        last_position[:] = [event.x, event.y]
        radius = brush_size // 2
        canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            fill="white",
            outline="white",
        )
        drawing.ellipse(
            (
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius,
            ),
            fill=255,
        )

    def continue_draw(event):
        old_x, old_y = last_position
        if old_x is None:
            return
        canvas.create_line(
            old_x,
            old_y,
            event.x,
            event.y,
            fill="white",
            width=brush_size,
            capstyle=tk.ROUND,
            smooth=True,
        )
        drawing.line((old_x, old_y, event.x, event.y), fill=255, width=brush_size)
        radius = brush_size // 2
        drawing.ellipse(
            (
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius,
            ),
            fill=255,
        )
        last_position[:] = [event.x, event.y]

    def stop_draw(_event=None):
        last_position[:] = [None, None]

    def clear():
        nonlocal drawing_image, drawing
        canvas.delete("all")
        drawing_image = Image.new("L", (size, size), 0)
        drawing = ImageDraw.Draw(drawing_image)
        stop_draw()

    def save_example(target_directory, label_name):
        prepared = prepare_image(drawing_image)
        if prepared is None:
            messagebox.showwarning("Пустой рисунок", "Сначала нарисуйте символ.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prepared.save(target_directory / f"{label_name}_{timestamp}.png")
        clear()
        update_counter()

    canvas.bind("<Button-1>", start_draw)
    canvas.bind("<B1-Motion>", continue_draw)
    canvas.bind("<ButtonRelease-1>", stop_draw)

    buttons = tk.Frame(root)
    buttons.pack(pady=8)

    tk.Button(
        buttons,
        text="Сохранить: НЕИЗВЕСТНЫЙ",
        command=lambda: save_example(CUSTOM_UNKNOWN_DIR, "unknown"),
        font=("Arial", 11, "bold"),
        fg="#b00020",
        width=25,
        height=2,
    ).pack(side=tk.LEFT, padx=4)

    tk.Button(
        buttons,
        text="Сохранить: ЦИФРА",
        command=lambda: save_example(CUSTOM_DIGITS_DIR, "digit"),
        font=("Arial", 11, "bold"),
        fg="#006b79",
        width=20,
        height=2,
    ).pack(side=tk.LEFT, padx=4)

    tk.Button(
        buttons,
        text="Очистить",
        command=clear,
        font=("Arial", 11),
        width=12,
        height=2,
    ).pack(side=tk.LEFT, padx=4)

    update_counter()
    root.mainloop()


# ============================================================
# НАБОРЫ ДАННЫХ
# ============================================================

def find_font_files():
    candidate_directories = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
    ]
    font_files = []

    for directory in candidate_directories:
        if directory.exists():
            font_files.extend(directory.glob("*.ttf"))
            font_files.extend(directory.glob("*.otf"))

    return font_files


SYMBOLS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЖЗИЙКЛМНПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдежзийклмнпрстуфхцчшщъыьэюя"
    "+-=<>!?@#$%&*()[]{}\\/|_^~:;,."
)


class SyntheticUnknownDataset(Dataset):
    def __init__(self, count, seed, font_files):
        self.count = count
        self.seed = seed
        self.font_files = font_files

    def __len__(self):
        return self.count

    def _load_font(self, rng, size):
        if not self.font_files:
            return ImageFont.load_default()

        font_path = rng.choice(self.font_files)
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            return ImageFont.load_default()

    def __getitem__(self, index):
        rng = random.Random(self.seed + index * 104729)
        canvas = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(canvas)
        mode = rng.random()

        if mode < 0.62:
            symbol = rng.choice(SYMBOLS)
            font = self._load_font(rng, rng.randint(28, 52))
            try:
                bbox = draw.textbbox((0, 0), symbol, font=font)
                width = max(1, bbox[2] - bbox[0])
                height = max(1, bbox[3] - bbox[1])
                x = (64 - width) // 2 - bbox[0] + rng.randint(-5, 5)
                y = (64 - height) // 2 - bbox[1] + rng.randint(-5, 5)
                draw.text((x, y), symbol, fill=255, font=font)
            except (OSError, UnicodeError):
                draw.line((10, 12, 54, 52), fill=255, width=rng.randint(3, 7))
                draw.line((54, 12, 10, 52), fill=255, width=rng.randint(3, 7))

        elif mode < 0.84:
            point_count = rng.randint(3, 7)
            points = [
                (rng.randint(8, 56), rng.randint(8, 56))
                for _ in range(point_count)
            ]
            draw.line(
                points,
                fill=255,
                width=rng.randint(3, 7),
                joint="curve",
            )
            if rng.random() < 0.45:
                draw.line(
                    (rng.randint(8, 56), rng.randint(8, 56),
                     rng.randint(8, 56), rng.randint(8, 56)),
                    fill=255,
                    width=rng.randint(2, 6),
                )

        else:
            shape = rng.choice(("triangle", "cross_box", "arrow", "grid"))
            width = rng.randint(3, 7)
            if shape == "triangle":
                draw.line((32, 7, 56, 55, 8, 55, 32, 7), fill=255, width=width)
            elif shape == "cross_box":
                draw.rectangle((10, 10, 54, 54), outline=255, width=width)
                draw.line((10, 10, 54, 54), fill=255, width=width)
            elif shape == "arrow":
                draw.line((8, 32, 54, 32), fill=255, width=width)
                draw.line((39, 16, 55, 32, 39, 48), fill=255, width=width)
            else:
                draw.line((8, 22, 56, 22), fill=255, width=width)
                draw.line((8, 42, 56, 42), fill=255, width=width)
                draw.line((22, 8, 22, 56), fill=255, width=width)
                draw.line((42, 8, 42, 56), fill=255, width=width)

        angle = rng.uniform(-18, 18)
        canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR)
        prepared = prepare_image(canvas)

        if prepared is None:
            prepared = Image.new("L", (28, 28), 0)
            ImageDraw.Draw(prepared).line((4, 4, 23, 23), fill=255, width=4)

        return ToTensor()(prepared), torch.tensor(0.0)


class CustomImageDataset(Dataset):
    def __init__(self, paths, label, repeat=1, augment=False):
        self.paths = list(paths)
        self.label = float(label)
        self.repeat = max(1, repeat)
        self.augment = transforms.RandomAffine(
            degrees=12,
            translate=(0.08, 0.08),
            scale=(0.90, 1.10),
            shear=7,
            fill=0,
        ) if augment else None

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, index):
        path = self.paths[index % len(self.paths)]
        with Image.open(path) as opened:
            image = opened.convert("L")

        prepared = prepare_image(image)
        if prepared is None:
            prepared = Image.new("L", (28, 28), 0)

        if self.augment is not None:
            prepared = self.augment(prepared)

        return ToTensor()(prepared), torch.tensor(self.label)


class LabeledDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = float(label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, torch.tensor(self.label)


def list_images(directory):
    directory.mkdir(parents=True, exist_ok=True)
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_custom_paths(paths):
    paths = list(paths)
    random.Random(RANDOM_SEED).shuffle(paths)

    if len(paths) < 5:
        return paths, []

    validation_count = max(1, round(len(paths) * 0.20))
    return paths[validation_count:], paths[:validation_count]


def load_mnist(train):
    transform = ToTensor()

    try:
        return datasets.MNIST(DATA_DIR, train=train, download=False, transform=transform)
    except RuntimeError:
        print("MNIST не найден локально. Выполняется первая загрузка...")
        return datasets.MNIST(DATA_DIR, train=train, download=True, transform=transform)


def build_datasets():
    mnist_train = load_mnist(train=True)
    mnist_test = load_mnist(train=False)
    font_files = find_font_files()

    custom_digit_train, custom_digit_validation = split_custom_paths(
        list_images(CUSTOM_DIGITS_DIR)
    )
    custom_unknown_train, custom_unknown_validation = split_custom_paths(
        list_images(CUSTOM_UNKNOWN_DIR)
    )

    training_parts = [
        LabeledDataset(mnist_train, 1),
        SyntheticUnknownDataset(
            SYNTHETIC_TRAIN_COUNT,
            RANDOM_SEED,
            font_files,
        ),
    ]

    if custom_digit_train:
        training_parts.append(
            CustomImageDataset(
                custom_digit_train,
                label=1,
                repeat=CUSTOM_TRAIN_REPEATS,
                augment=True,
            )
        )

    if custom_unknown_train:
        training_parts.append(
            CustomImageDataset(
                custom_unknown_train,
                label=0,
                repeat=CUSTOM_TRAIN_REPEATS,
                augment=True,
            )
        )

    validation_parts = [
        LabeledDataset(mnist_test, 1),
        SyntheticUnknownDataset(
            SYNTHETIC_VALIDATION_COUNT,
            RANDOM_SEED + 2_000_000,
            font_files,
        ),
    ]

    if custom_digit_validation:
        validation_parts.append(
            CustomImageDataset(custom_digit_validation, label=1)
        )

    if custom_unknown_validation:
        validation_parts.append(
            CustomImageDataset(custom_unknown_validation, label=0)
        )

    metadata = {
        "custom_digit_train": len(custom_digit_train),
        "custom_digit_validation": len(custom_digit_validation),
        "custom_unknown_train": len(custom_unknown_train),
        "custom_unknown_validation": len(custom_unknown_validation),
        "font_count": len(font_files),
    }

    return ConcatDataset(training_parts), ConcatDataset(validation_parts), metadata


# ============================================================
# ОБУЧЕНИЕ И ПРОВЕРКА
# ============================================================

def set_random_seeds():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def evaluate(model, loader, device):
    model.eval()
    true_digits = 0
    accepted_digits = 0
    true_unknown = 0
    rejected_unknown = 0
    total_loss = 0.0
    total_count = 0
    loss_function = nn.BCEWithLogitsLoss(reduction="sum")

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= DIGIT_THRESHOLD

            total_loss += loss_function(logits, labels).item()
            total_count += labels.numel()

            digit_mask = labels >= 0.5
            unknown_mask = ~digit_mask
            true_digits += digit_mask.sum().item()
            accepted_digits += (predictions & digit_mask).sum().item()
            true_unknown += unknown_mask.sum().item()
            rejected_unknown += ((~predictions) & unknown_mask).sum().item()

    digit_recall = accepted_digits / max(1, true_digits)
    unknown_recall = rejected_unknown / max(1, true_unknown)
    balanced_accuracy = (digit_recall + unknown_recall) / 2

    return {
        "loss": total_loss / max(1, total_count),
        "digit_recall": digit_recall,
        "unknown_recall": unknown_recall,
        "balanced_accuracy": balanced_accuracy,
    }


def backup_existing_model():
    if not MODEL_FILE.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MODEL_FILE.with_name(f"unknown_detector_backup_{timestamp}.pth")
    shutil.copy2(MODEL_FILE, backup_path)
    return backup_path


def train():
    set_random_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_dataset, validation_dataset, metadata = build_datasets()

    batch_size = 256 if device.type == "cuda" else 128
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = UnknownDetector().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )
    loss_function = nn.BCEWithLogitsLoss()

    print("Устройство:", device)
    print("Обучающих примеров:", len(training_dataset))
    print("Проверочных примеров:", len(validation_dataset))
    print("Собственных цифр:", metadata["custom_digit_train"] + metadata["custom_digit_validation"])
    print("Собственных неизвестных:", metadata["custom_unknown_train"] + metadata["custom_unknown_validation"])
    print(f"Рабочий порог: {DIGIT_THRESHOLD:.0%}")

    best_state = None
    best_metrics = None
    best_epoch = 0
    best_score = -math.inf

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        example_count = 0

        for images, labels in training_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.numel()
            example_count += labels.numel()

        scheduler.step()
        metrics = evaluate(model, validation_loader, device)

        # Особенно строго защищаем настоящие цифры от ложного отклонения.
        digit_penalty = max(0.0, 0.985 - metrics["digit_recall"]) * 5
        score = metrics["balanced_accuracy"] - digit_penalty

        print(
            f"Эпоха {epoch}/{EPOCHS}: "
            f"train loss {running_loss / max(1, example_count):.4f}; "
            f"digit {metrics['digit_recall']:.2%}; "
            f"unknown {metrics['unknown_recall']:.2%}; "
            f"balanced {metrics['balanced_accuracy']:.2%}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("Не удалось выбрать обученную модель.")

    backup_path = backup_existing_model()
    torch.save(best_state, MODEL_FILE)

    config = {
        "digit_probability_threshold": DIGIT_THRESHOLD,
        "architecture": "UnknownDetector_v1",
        "best_epoch": best_epoch,
        "digit_recall_percent": round(best_metrics["digit_recall"] * 100, 4),
        "unknown_recall_percent": round(best_metrics["unknown_recall"] * 100, 4),
        "balanced_accuracy_percent": round(
            best_metrics["balanced_accuracy"] * 100,
            4,
        ),
        "custom_digit_count": (
            metadata["custom_digit_train"] + metadata["custom_digit_validation"]
        ),
        "custom_unknown_count": (
            metadata["custom_unknown_train"] + metadata["custom_unknown_validation"]
        ),
    }

    with CONFIG_FILE.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    report_lines = [
        "ОТЧЁТ ОБ ОБУЧЕНИИ ДЕТЕКТОРА НЕИЗВЕСТНЫХ СИМВОЛОВ",
        "",
        f"Выбрана эпоха: {best_epoch}",
        f"Порог принятия цифры: {DIGIT_THRESHOLD:.0%}",
        f"Сохранено настоящих цифр: {best_metrics['digit_recall']:.2%}",
        f"Обнаружено неизвестных символов: {best_metrics['unknown_recall']:.2%}",
        f"Сбалансированная точность: {best_metrics['balanced_accuracy']:.2%}",
        f"Собственных примеров цифр: {config['custom_digit_count']}",
        f"Собственных неизвестных символов: {config['custom_unknown_count']}",
        f"Синтетических неизвестных для обучения: {SYNTHETIC_TRAIN_COUNT}",
        f"Количество найденных шрифтов: {metadata['font_count']}",
        "",
        f"Модель: {MODEL_FILE.name}",
        f"Конфигурация: {CONFIG_FILE.name}",
    ]

    if backup_path is not None:
        report_lines.append(f"Резервная копия прежней модели: {backup_path.name}")

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nОбучение завершено.")
    print("Создано:", MODEL_FILE.name)
    print("Создано:", CONFIG_FILE.name)
    print("Создано:", REPORT_FILE.name)
    if backup_path is not None:
        print("Старая модель сохранена как:", backup_path.name)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Обучение детектора: цифра или неизвестный символ"
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="открыть окно для сбора своих примеров",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.collect:
        run_collector()
    else:
        train()
