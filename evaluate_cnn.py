from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


DATA_DIR = Path("data")
BASE_MODEL_FILE = Path("mnist_cnn_model.pth")
USER_MODEL_FILE = Path("mnist_cnn_user.pth")
REPORT_FILE = Path("cnn_evaluation_report.txt")
BATCH_SIZE = 256


class CNN(nn.Module):
    """Та же архитектура, которая использовалась при обучении."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def evaluate_model(model_file, test_loader, device):
    model = CNN().to(device)
    state = torch.load(
        model_file,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    # Строка = правильная цифра, столбец = ответ модели.
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    total = 0
    correct = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels_device = labels.to(
                device,
                non_blocking=device.type == "cuda",
            )

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels_device).sum().item()
            total += labels.size(0)

            pairs = labels * 10 + predictions.cpu()
            confusion += torch.bincount(
                pairs,
                minlength=100,
            ).reshape(10, 10)

    accuracy = 100.0 * correct / total
    return accuracy, confusion


def build_report(model_name, model_file, accuracy, confusion):
    lines = [
        "=" * 72,
        f"Модель: {model_name}",
        f"Файл: {model_file}",
        f"Общая точность: {accuracy:.2f}%",
        "",
        "Точность по каждой цифре:",
    ]

    for digit in range(10):
        total = confusion[digit].sum().item()
        correct = confusion[digit, digit].item()
        class_accuracy = 100.0 * correct / total
        lines.append(
            f"  {digit}: {class_accuracy:6.2f}%  "
            f"({correct}/{total})"
        )

    lines.extend(
        [
            "",
            "Матрица ошибок:",
            "Строка = правильная цифра, столбец = ответ модели.",
            "",
            "прав.\\ответ" + "".join(f"{digit:>6}" for digit in range(10)),
        ]
    )

    for digit in range(10):
        row = "".join(
            f"{confusion[digit, predicted].item():>6}"
            for predicted in range(10)
        )
        lines.append(f"{digit:>11}{row}")

    errors = []
    for correct_digit in range(10):
        for predicted_digit in range(10):
            if correct_digit == predicted_digit:
                continue

            count = confusion[correct_digit, predicted_digit].item()
            if count > 0:
                errors.append((count, correct_digit, predicted_digit))

    errors.sort(reverse=True)

    lines.extend(["", "Самые частые ошибки:"])
    for count, correct_digit, predicted_digit in errors[:10]:
        lines.append(
            f"  цифра {correct_digit} распознана как "
            f"{predicted_digit}: {count} раз"
        )

    return lines


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Устройство: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not BASE_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Не найден файл {BASE_MODEL_FILE}. "
            "Поместите evaluate_cnn.py в папку C:\\AI."
        )

    test_data = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    models = [("Базовая CNN", BASE_MODEL_FILE)]

    if USER_MODEL_FILE.exists():
        models.append(
            ("CNN после пользовательских исправлений", USER_MODEL_FILE)
        )
    else:
        print(
            f"Файл {USER_MODEL_FILE} пока не создан — "
            "проверяется только базовая CNN."
        )

    full_report = [
        "ОЦЕНКА CNN НА ТЕСТОВОМ НАБОРЕ MNIST",
        f"Количество изображений: {len(test_data)}",
        f"Устройство: {device}",
        "",
    ]

    results = []

    for model_name, model_file in models:
        print(f"Проверяю: {model_name}...")
        accuracy, confusion = evaluate_model(
            model_file,
            test_loader,
            device,
        )
        results.append((model_name, accuracy))
        full_report.extend(
            build_report(
                model_name,
                model_file,
                accuracy,
                confusion,
            )
        )
        full_report.append("")

    if len(results) == 2:
        difference = results[1][1] - results[0][1]
        full_report.extend(
            [
                "=" * 72,
                "СРАВНЕНИЕ МОДЕЛЕЙ",
                f"Базовая CNN: {results[0][1]:.2f}%",
                f"CNN после исправлений: {results[1][1]:.2f}%",
                f"Изменение: {difference:+.2f} процентного пункта",
                "",
                (
                    "Небольшое изменение нормально. Сильное падение может "
                    "означать, что дообучение на пользовательских примерах "
                    "ухудшило знания модели о MNIST."
                ),
            ]
        )

    report_text = "\n".join(full_report)
    print()
    print(report_text)

    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print()
    print(f"Отчёт сохранён: {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()

