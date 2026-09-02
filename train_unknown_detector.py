import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import RandomAffine, ToTensor


# ============================================================
# НАСТРОЙКИ
# ============================================================

DATA_DIR = "data"
CUSTOM_UNKNOWN_DIR = "unknown_symbols"
MODEL_FILE = "unknown_detector.pth"
CONFIG_FILE = "unknown_detector_config.json"
REPORT_FILE = "unknown_detector_report.txt"

SEED = 20260902
EPOCHS = 4
BATCH_SIZE = 128
LEARNING_RATE = 0.0003
TARGET_DIGIT_RECALL = 0.995

MNIST_TRAIN_COUNT = 50000
MNIST_VALIDATION_COUNT = 10000
EMNIST_TRAIN_COUNT = 30000
EMNIST_VALIDATION_COUNT = 7000
SYNTHETIC_TEXT_TRAIN_WITH_EMNIST = 10000
SYNTHETIC_TEXT_VALIDATION_WITH_EMNIST = 1500
SYNTHETIC_TEXT_TRAIN_WITHOUT_EMNIST = 35000
SYNTHETIC_TEXT_VALIDATION_WITHOUT_EMNIST = 6500
SYNTHETIC_SYMBOL_TRAIN_WITH_EMNIST = 10000
SYNTHETIC_SYMBOL_VALIDATION_WITH_EMNIST = 1500
SYNTHETIC_SYMBOL_TRAIN_WITHOUT_EMNIST = 15000
SYNTHETIC_SYMBOL_VALIDATION_WITHOUT_EMNIST = 3500
MAX_CUSTOM_REPEATS = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# МОДЕЛЬ: ЦИФРА (1) / НЕИЗВЕСТНЫЙ СИМВОЛ (0)
# ============================================================

class UnknownDetector(nn.Module):

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
# PREPROCESSING, СОВПАДАЮЩИЙ С ОСНОВНОЙ ПРОГРАММОЙ
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

    center_x = sum_x / total
    center_y = sum_y / total
    shift_x = round(13.5 - center_x)
    shift_y = round(13.5 - center_y)

    centered = Image.new("L", (28, 28), 0)
    centered.paste(image, (shift_x, shift_y))
    return centered


def prepare_symbol(image):

    image = ImageOps.grayscale(image)
    bbox = image.getbbox()

    if bbox is None:
        return Image.new("L", (28, 28), 0)

    image = image.crop(bbox)
    width, height = image.size
    scale = 20 / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    prepared = Image.new("L", (28, 28), 0)
    prepared.paste(
        image,
        ((28 - new_width) // 2, (28 - new_height) // 2)
    )
    return center_by_mass(prepared)


train_affine = RandomAffine(
    degrees=12,
    translate=(0.10, 0.10),
    scale=(0.86, 1.14),
    shear=7,
    fill=0,
)


class PreparedDataset(Dataset):

    def __init__(self, base_dataset, label, augment=False, fix_emnist=False):
        self.base_dataset = base_dataset
        self.label = float(label)
        self.augment = augment
        self.fix_emnist = fix_emnist

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]

        if self.fix_emnist:
            # torchvision хранит EMNIST в транспонированной ориентации.
            image = image.transpose(Image.Transpose.ROTATE_90)
            image = ImageOps.mirror(image)

        if self.augment:
            image = train_affine(image)

        image = prepare_symbol(image)
        return ToTensor()(image), torch.tensor(self.label, dtype=torch.float32)


def find_font_files():

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/seguisb.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/cour.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ]
    return [path for path in candidates if path.exists()]


FONT_FILES = find_font_files()
UNKNOWN_CHARACTERS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "@#$&*+=!?<>[]{}()/%"
)


class SyntheticTextUnknownDataset(Dataset):
    """Буквы и знаки, созданные локально без загрузки из интернета."""

    def __init__(self, count, seed_offset, augment=False):
        self.count = count
        self.seed_offset = seed_offset
        self.augment = augment

    def __len__(self):
        return self.count

    def _font(self, rng):
        size = rng.randint(34, 54)

        if FONT_FILES:
            return ImageFont.truetype(str(rng.choice(FONT_FILES)), size=size)

        return ImageFont.load_default()

    def __getitem__(self, index):
        rng = random.Random(SEED + self.seed_offset + index)
        image = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(image)
        character = UNKNOWN_CHARACTERS[index % len(UNKNOWN_CHARACTERS)]
        font = self._font(rng)
        stroke_width = rng.randint(0, 2)

        bbox = draw.textbbox(
            (0, 0),
            character,
            font=font,
            stroke_width=stroke_width,
        )
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        x = (64 - width) // 2 - bbox[0] + rng.randint(-5, 5)
        y = (64 - height) // 2 - bbox[1] + rng.randint(-5, 5)

        draw.text(
            (x, y),
            character,
            font=font,
            fill=255,
            stroke_width=stroke_width,
            stroke_fill=255,
        )

        if self.augment:
            image = train_affine(image)

        return ToTensor()(prepare_symbol(image)), torch.tensor(0.0)


class SyntheticUnknownDataset(Dataset):

    def __init__(self, count, seed_offset, augment=False):
        self.count = count
        self.seed_offset = seed_offset
        self.augment = augment

    def __len__(self):
        return self.count

    @staticmethod
    def _point(rng):
        return rng.randint(8, 55), rng.randint(8, 55)

    def __getitem__(self, index):
        rng = random.Random(SEED + self.seed_offset + index)
        image = Image.new("L", (64, 64), 0)
        draw = ImageDraw.Draw(image)
        width = rng.randint(4, 8)
        kind = index % 9

        if kind == 0:  # плюс
            cx, cy = self._point(rng)
            size = rng.randint(13, 23)
            draw.line((cx - size, cy, cx + size, cy), fill=255, width=width)
            draw.line((cx, cy - size, cx, cy + size), fill=255, width=width)
        elif kind == 1:  # крест
            x1, y1 = self._point(rng)
            x2, y2 = self._point(rng)
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            if right - left < 25:
                right = min(58, left + 30)
            if bottom - top < 25:
                bottom = min(58, top + 30)
            draw.line((left, top, right, bottom), fill=255, width=width)
            draw.line((left, bottom, right, top), fill=255, width=width)
        elif kind == 2:  # решётка
            x = rng.randint(14, 24)
            y = rng.randint(14, 24)
            size = rng.randint(24, 34)
            draw.line((x + 7, y, x + 3, y + size), fill=255, width=width)
            draw.line((x + 20, y, x + 16, y + size), fill=255, width=width)
            draw.line((x, y + 10, x + size, y + 7), fill=255, width=width)
            draw.line((x, y + 23, x + size, y + 20), fill=255, width=width)
        elif kind == 3:  # знак равенства
            x = rng.randint(8, 18)
            y = rng.randint(18, 28)
            length = rng.randint(30, 45)
            gap = rng.randint(10, 16)
            draw.line((x, y, x + length, y), fill=255, width=width)
            draw.line((x, y + gap, x + length, y + gap), fill=255, width=width)
        elif kind == 4:  # треугольник с дополнительным штрихом
            p1 = (rng.randint(25, 35), rng.randint(5, 12))
            p2 = (rng.randint(5, 15), rng.randint(48, 58))
            p3 = (rng.randint(48, 58), rng.randint(48, 58))
            draw.line((p1, p2, p3, p1), fill=255, width=width, joint="curve")
            draw.line((p2[0] + 5, 38, p3[0] - 5, 38), fill=255, width=width)
        elif kind == 5:  # квадрат с диагональю
            x = rng.randint(8, 16)
            y = rng.randint(8, 16)
            size = rng.randint(34, 45)
            draw.rectangle((x, y, x + size, y + size), outline=255, width=width)
            draw.line((x, y + size, x + size, y), fill=255, width=width)
        elif kind == 6:  # проценты-подобный знак
            draw.ellipse((8, 8, 23, 23), outline=255, width=max(2, width - 2))
            draw.line((18, 50, 47, 12), fill=255, width=width)
            draw.ellipse((41, 40, 56, 55), outline=255, width=max(2, width - 2))
        elif kind == 7:  # зигзаг с отдельным штрихом
            points = [
                (rng.randint(6, 13), rng.randint(10, 20)),
                (rng.randint(24, 31), rng.randint(42, 55)),
                (rng.randint(35, 43), rng.randint(9, 20)),
                (rng.randint(50, 58), rng.randint(42, 55)),
            ]
            draw.line(points, fill=255, width=width, joint="curve")
            draw.line((8, 55, 32, 55), fill=255, width=width)
        else:  # случайная многолинейная каракуля
            points = [self._point(rng) for _ in range(rng.randint(4, 7))]
            draw.line(points, fill=255, width=width, joint="curve")
            a = self._point(rng)
            b = self._point(rng)
            draw.line((a, b), fill=255, width=width)

        if self.augment:
            image = train_affine(image)

        return ToTensor()(prepare_symbol(image)), torch.tensor(0.0)


class CustomUnknownDataset(Dataset):

    def __init__(self, directory, repeats=1, augment=True):
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        self.files = sorted(
            path
            for path in Path(directory).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        self.repeats = repeats
        self.augment = augment

    def __len__(self):
        return len(self.files) * self.repeats

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]

        with Image.open(path) as source:
            image = source.convert("L")

        if self.augment:
            image = train_affine(image)

        return ToTensor()(prepare_symbol(image)), torch.tensor(0.0)


# ============================================================
# ОБУЧЕНИЕ И КАЛИБРОВКА ПОРОГА
# ============================================================

def fixed_subset(dataset, count, seed):
    count = min(count, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def try_load_emnist():

    try:
        print("Пробуем загрузить EMNIST Letters...")
        train_set = datasets.EMNIST(
            DATA_DIR,
            split="letters",
            train=True,
            download=True,
        )
        test_set = datasets.EMNIST(
            DATA_DIR,
            split="letters",
            train=False,
            download=True,
        )
        print("EMNIST Letters доступен.")
        return train_set, test_set, True
    except Exception as error:
        print()
        print("ПРЕДУПРЕЖДЕНИЕ: EMNIST загрузить не удалось.")
        print("Причина:", error)
        print(
            "Обучение продолжится безопасно без отключения SSL. "
            "EMNIST заменён синтетическими буквами и знаками из шрифтов Windows."
        )
        print()
        return None, None, False


def make_datasets():

    print("Загрузка MNIST...")

    mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True)
    emnist_train, emnist_test, emnist_available = try_load_emnist()

    generator = torch.Generator().manual_seed(SEED)
    mnist_order = torch.randperm(len(mnist_train), generator=generator).tolist()
    train_indices = mnist_order[:MNIST_TRAIN_COUNT]
    validation_indices = mnist_order[
        MNIST_TRAIN_COUNT:MNIST_TRAIN_COUNT + MNIST_VALIDATION_COUNT
    ]

    positive_train = PreparedDataset(
        Subset(mnist_train, train_indices),
        label=1,
        augment=True,
    )
    positive_validation = PreparedDataset(
        Subset(mnist_train, validation_indices),
        label=1,
        augment=False,
    )

    if emnist_available:
        negative_train_parts = [
            PreparedDataset(
                fixed_subset(emnist_train, EMNIST_TRAIN_COUNT, SEED + 1),
                label=0,
                augment=True,
                fix_emnist=True,
            ),
            SyntheticTextUnknownDataset(
                SYNTHETIC_TEXT_TRAIN_WITH_EMNIST,
                seed_offset=50000,
                augment=True,
            ),
            SyntheticUnknownDataset(
                SYNTHETIC_SYMBOL_TRAIN_WITH_EMNIST,
                seed_offset=100000,
                augment=True,
            ),
        ]
        negative_validation_parts = [
            PreparedDataset(
                fixed_subset(emnist_test, EMNIST_VALIDATION_COUNT, SEED + 3),
                label=0,
                augment=False,
                fix_emnist=True,
            ),
            SyntheticTextUnknownDataset(
                SYNTHETIC_TEXT_VALIDATION_WITH_EMNIST,
                seed_offset=150000,
                augment=False,
            ),
            SyntheticUnknownDataset(
                SYNTHETIC_SYMBOL_VALIDATION_WITH_EMNIST,
                seed_offset=200000,
                augment=False,
            ),
        ]
    else:
        negative_train_parts = [
            SyntheticTextUnknownDataset(
                SYNTHETIC_TEXT_TRAIN_WITHOUT_EMNIST,
                seed_offset=50000,
                augment=True,
            ),
            SyntheticUnknownDataset(
                SYNTHETIC_SYMBOL_TRAIN_WITHOUT_EMNIST,
                seed_offset=100000,
                augment=True,
            ),
        ]
        negative_validation_parts = [
            SyntheticTextUnknownDataset(
                SYNTHETIC_TEXT_VALIDATION_WITHOUT_EMNIST,
                seed_offset=150000,
                augment=False,
            ),
            SyntheticUnknownDataset(
                SYNTHETIC_SYMBOL_VALIDATION_WITHOUT_EMNIST,
                seed_offset=200000,
                augment=False,
            ),
        ]

    custom_dataset = CustomUnknownDataset(
        CUSTOM_UNKNOWN_DIR,
        repeats=1,
        augment=True,
    )
    custom_file_count = len(custom_dataset.files)

    if custom_file_count:
        repeats = min(
            MAX_CUSTOM_REPEATS,
            max(5, math.ceil(3000 / custom_file_count))
        )
        negative_train_parts.append(
            CustomUnknownDataset(
                CUSTOM_UNKNOWN_DIR,
                repeats=repeats,
                augment=True,
            )
        )
        print(
            f"Пользовательских неизвестных символов: {custom_file_count}; "
            f"повторений с аугментацией: {repeats}."
        )
    else:
        Path(CUSTOM_UNKNOWN_DIR).mkdir(parents=True, exist_ok=True)
        print(
            f"Папка {CUSTOM_UNKNOWN_DIR} пока пуста — используется общий набор."
        )

    negative_train = ConcatDataset(negative_train_parts)

    negative_validation = ConcatDataset(negative_validation_parts)

    training = ConcatDataset([positive_train, negative_train])
    return (
        training,
        positive_validation,
        negative_validation,
        custom_file_count,
        emnist_available,
    )


def collect_probabilities(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    probabilities = []

    model.eval()

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            batch_probabilities = torch.sigmoid(model(images))
            probabilities.append(batch_probabilities.cpu())

    return torch.cat(probabilities)


def calibrate(model, positive_validation, negative_validation):

    digit_probabilities = collect_probabilities(model, positive_validation)
    unknown_probabilities = collect_probabilities(model, negative_validation)

    reject_fraction = 1.0 - TARGET_DIGIT_RECALL
    threshold = float(torch.quantile(digit_probabilities, reject_fraction).item())
    threshold = max(0.01, min(threshold, 0.99))

    digit_recall = float((digit_probabilities >= threshold).float().mean().item())
    unknown_recall = float((unknown_probabilities < threshold).float().mean().item())

    return {
        "threshold": threshold,
        "digit_recall": digit_recall,
        "unknown_recall": unknown_recall,
        "digit_probability_mean": float(digit_probabilities.mean().item()),
        "unknown_probability_mean": float(unknown_probabilities.mean().item()),
    }


def train():

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    (
        training,
        positive_validation,
        negative_validation,
        custom_count,
        emnist_available,
    ) = make_datasets()

    train_loader = DataLoader(
        training,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = UnknownDetector().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )
    criterion = nn.BCEWithLogitsLoss()

    print("Устройство:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Обучающих примеров:", len(training))
    print("Цель: сохранить не менее 99.5% настоящих цифр.")

    best_state = None
    best_metrics = None
    best_epoch = 0
    history_lines = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

        metrics = calibrate(model, positive_validation, negative_validation)
        average_loss = running_loss / max(1, total)

        line = (
            f"Эпоха {epoch}/{EPOCHS}: loss={average_loss:.5f}; "
            f"сохранено цифр={metrics['digit_recall'] * 100:.2f}%; "
            f"найдено неизвестных={metrics['unknown_recall'] * 100:.2f}%; "
            f"порог={metrics['threshold']:.4f}"
        )
        print(line)
        history_lines.append(line)

        if (
            best_metrics is None
            or metrics["unknown_recall"] > best_metrics["unknown_recall"]
        ):
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None or best_metrics is None:
        raise RuntimeError("Не удалось выбрать модель.")

    torch.save(best_state, MODEL_FILE)

    config = {
        "format_version": 1,
        "model_type": "binary_digit_unknown_detector",
        "digit_probability_threshold": best_metrics["threshold"],
        "target_digit_recall": TARGET_DIGIT_RECALL,
        "validation_digit_recall": best_metrics["digit_recall"],
        "validation_unknown_recall": best_metrics["unknown_recall"],
        "best_epoch": best_epoch,
        "custom_unknown_examples": custom_count,
        "emnist_used": emnist_available,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)

    report_lines = [
        "ОТЧЁТ: ДЕТЕКТОР НЕИЗВЕСТНЫХ СИМВОЛОВ",
        "=" * 52,
        f"Устройство: {device}",
        f"Выбранная эпоха: {best_epoch}",
        f"Порог 'это цифра': {best_metrics['threshold']:.6f}",
        f"Сохранено настоящих цифр: {best_metrics['digit_recall'] * 100:.2f}%",
        f"Найдено неизвестных символов: {best_metrics['unknown_recall'] * 100:.2f}%",
        f"Средняя вероятность для цифр: {best_metrics['digit_probability_mean'] * 100:.2f}%",
        f"Средняя вероятность для неизвестных: {best_metrics['unknown_probability_mean'] * 100:.2f}%",
        f"Пользовательских неизвестных примеров: {custom_count}",
        f"EMNIST использован: {'да' if emnist_available else 'нет (безопасная локальная замена)'}",
        "",
        "История:",
        *history_lines,
        "",
        f"Сохранено: {MODEL_FILE}",
        f"Сохранено: {CONFIG_FILE}",
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))

    print()
    print("Готово.")
    print(f"Модель: {MODEL_FILE}")
    print(f"Конфигурация: {CONFIG_FILE}")
    print(f"Отчёт: {REPORT_FILE}")


if __name__ == "__main__":
    try:
        train()
    except Exception as error:
        print()
        print("ОШИБКА:", error)
        print(
            "Проверьте наличие MNIST в папке data и torch/torchvision "
            "в активированном .venv. Ошибка EMNIST отдельно обрабатывается "
            "и не должна останавливать обучение."
        )
        raise
