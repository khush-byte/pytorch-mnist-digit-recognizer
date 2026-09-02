import csv
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# ============================================================
# НАСТРОЙКИ
# ============================================================

SEED = 42
EPOCHS = 12

# Свёрточные слои меняем медленнее классификатора: они уже хорошо
# выделяют универсальные линии, края и формы цифр.
FEATURE_LEARNING_RATE = 0.000005
CLASSIFIER_LEARNING_RATE = 0.00001

BATCH_SIZE = 64
USER_SAMPLES_PER_BATCH = 8
STEPS_PER_EPOCH = 80
MNIST_EVALUATION_BATCH_SIZE = 256

# Teacher-student distillation удерживает ответы новой модели рядом
# с ответами исходной CNN на MNIST replay-примерах.
DISTILLATION_TEMPERATURE = 2.0
DISTILLATION_WEIGHT = 0.50

# L2-SP: не даём весам слишком далеко отходить от исходной CNN.
WEIGHT_ANCHOR_STRENGTH = 0.10

# Новая модель не принимается, если MNIST ухудшился более чем на 0.15%.
# Это максимум 15 дополнительных ошибок на 10 000 изображений.
MAX_MNIST_DROP = 0.15

# Не допускаем падения отдельного класса пользовательской validation
# более чем на 5 процентных пунктов относительно выбранной исходной модели.
MAX_USER_CLASS_DROP = 5.0

# Останавливаем обучение, если несколько эпох подряд не дают
# нового безопасного лучшего результата.
EARLY_STOPPING_PATIENCE = 4

BASE_MODEL_FILE = Path("mnist_cnn_model.pth")
USER_MODEL_FILE = Path("mnist_cnn_user.pth")
CORRECTIONS_DIR = Path("corrections")

# Этот набор уже использовался для анализа и выбора направления улучшения,
# поэтому теперь это validation, а не финальный test.
USER_VALIDATION_DIR = Path("user_test_dataset")

DATA_DIR = Path("data")
VERSIONS_DIR = Path("model_versions")
HISTORY_FILE = Path("safe_finetune_history.csv")
REPORT_FILE = Path("safe_finetune_report.txt")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# АРХИТЕКТУРА CNN
# ============================================================

class CNN(nn.Module):

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


user_augmentation = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees=8,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=4,
            fill=0,
        ),
        transforms.ToTensor(),
    ]
)


# ============================================================
# ДАННЫЕ
# ============================================================

def load_labeled_paths(root_directory):
    samples = []

    for digit in range(10):
        directory = root_directory / str(digit)

        if not directory.is_dir():
            continue

        for path in sorted(directory.glob("*.png")):
            samples.append((path, digit))

    return samples


def load_user_tensor(path, augment):
    with Image.open(path) as opened:
        image = opened.convert("L").copy()

    if augment:
        return user_augmentation(image)

    return ToTensor()(image)


def copy_state_dict_to_cpu(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


# ============================================================
# ОЦЕНКА
# ============================================================

@torch.inference_mode()
def evaluate_mnist(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )

        labels = labels.to(
            device,
            non_blocking=device.type == "cuda",
        )

        predictions = model(images).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


@torch.inference_mode()
def evaluate_user_validation(model, samples):
    model.eval()

    confusion = torch.zeros(10, 10, dtype=torch.int64)
    batch_size = 128

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]

        images = torch.stack(
            [
                load_user_tensor(path, augment=False)
                for path, _ in batch_samples
            ]
        ).to(device)

        labels_cpu = torch.tensor(
            [label for _, label in batch_samples],
            dtype=torch.long,
        )

        labels_device = labels_cpu.to(device)
        predictions = model(images).argmax(dim=1)

        pairs = labels_cpu * 10 + predictions.cpu()

        confusion += torch.bincount(
            pairs,
            minlength=100,
        ).reshape(10, 10)

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    accuracy = 100.0 * correct / total

    class_accuracy = {}

    for digit in range(10):
        class_total = confusion[digit].sum().item()

        if class_total == 0:
            class_accuracy[digit] = None
        else:
            class_accuracy[digit] = (
                100.0
                * confusion[digit, digit].item()
                / class_total
            )

    return accuracy, class_accuracy, confusion


def evaluate_state(state, mnist_loader, user_validation_samples):
    model = CNN().to(device)
    model.load_state_dict(state)

    mnist_accuracy = evaluate_mnist(model, mnist_loader)

    (
        user_accuracy,
        class_accuracy,
        confusion,
    ) = evaluate_user_validation(
        model,
        user_validation_samples,
    )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return mnist_accuracy, user_accuracy, class_accuracy, confusion


# ============================================================
# БЕЗОПАСНОСТЬ И ВЫБОР МОДЕЛИ
# ============================================================

def maximum_class_drop(reference, candidate):
    drops = []

    for digit in range(10):
        reference_value = reference[digit]
        candidate_value = candidate[digit]

        if reference_value is None or candidate_value is None:
            continue

        drops.append(reference_value - candidate_value)

    return max(drops, default=0.0)


def is_better_model(
    candidate_user_accuracy,
    candidate_mnist_accuracy,
    best_user_accuracy,
    best_mnist_accuracy,
):
    if candidate_user_accuracy > best_user_accuracy + 1e-9:
        return True

    if (
        abs(candidate_user_accuracy - best_user_accuracy) <= 1e-9
        and candidate_mnist_accuracy > best_mnist_accuracy + 1e-9
    ):
        return True

    return False


def format_class_accuracy(class_accuracy):
    parts = []

    for digit in range(10):
        value = class_accuracy[digit]

        if value is None:
            parts.append(f"{digit}: n/a")
        else:
            parts.append(f"{digit}: {value:.2f}%")

    return " | ".join(parts)


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train_one_epoch(
    model,
    teacher_model,
    optimizer,
    loss_fn,
    correction_samples,
    mnist_train_data,
    steps_per_epoch,
    anchor_parameters,
):
    model.train()
    teacher_model.eval()

    user_per_batch = min(
        USER_SAMPLES_PER_BATCH,
        max(1, len(correction_samples)),
    )

    replay_count = BATCH_SIZE - user_per_batch
    user_fraction = user_per_batch / BATCH_SIZE
    replay_fraction = replay_count / BATCH_SIZE

    loss_totals = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "distillation_loss": 0.0,
        "anchor_loss": 0.0,
    }

    for _ in range(steps_per_epoch):
        selected_user_samples = random.choices(
            correction_samples,
            k=user_per_batch,
        )

        user_images = []
        user_labels = []

        for path, label in selected_user_samples:
            user_images.append(
                load_user_tensor(
                    path,
                    augment=random.random() < 0.80,
                )
            )
            user_labels.append(label)

        replay_indexes = random.sample(
            range(len(mnist_train_data)),
            replay_count,
        )

        replay_images = []
        replay_labels = []

        for index in replay_indexes:
            image, label = mnist_train_data[index]
            replay_images.append(image)
            replay_labels.append(label)

        user_images = torch.stack(user_images).to(device)
        replay_images = torch.stack(replay_images).to(device)

        user_labels = torch.tensor(
            user_labels,
            dtype=torch.long,
            device=device,
        )

        replay_labels = torch.tensor(
            replay_labels,
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        user_logits = model(user_images)
        replay_logits = model(replay_images)

        user_loss = loss_fn(user_logits, user_labels)
        replay_loss = loss_fn(replay_logits, replay_labels)

        classification_loss = (
            user_fraction * user_loss
            + replay_fraction * replay_loss
        )

        with torch.no_grad():
            teacher_logits = teacher_model(replay_images)

        temperature = DISTILLATION_TEMPERATURE

        distillation_loss = nn.functional.kl_div(
            nn.functional.log_softmax(
                replay_logits / temperature,
                dim=1,
            ),
            nn.functional.softmax(
                teacher_logits / temperature,
                dim=1,
            ),
            reduction="batchmean",
        ) * (temperature ** 2)

        anchor_loss = torch.zeros((), device=device)

        for name, parameter in model.named_parameters():
            anchor_loss = anchor_loss + torch.sum(
                (parameter - anchor_parameters[name]) ** 2
            )

        loss = (
            classification_loss
            + DISTILLATION_WEIGHT * distillation_loss
            + WEIGHT_ANCHOR_STRENGTH * anchor_loss
        )

        loss.backward()
        optimizer.step()

        loss_totals["total_loss"] += loss.item()
        loss_totals["classification_loss"] += (
            classification_loss.item()
        )
        loss_totals["distillation_loss"] += (
            distillation_loss.item()
        )
        loss_totals["anchor_loss"] += anchor_loss.item()

    return {
        name: value / steps_per_epoch
        for name, value in loss_totals.items()
    }


def main():
    print(f"Устройство: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not BASE_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Не найден файл {BASE_MODEL_FILE}."
        )

    correction_samples = load_labeled_paths(CORRECTIONS_DIR)
    validation_samples = load_labeled_paths(USER_VALIDATION_DIR)

    if not correction_samples:
        raise RuntimeError(
            "Папка corrections не содержит обучающих изображений."
        )

    if not validation_samples:
        raise RuntimeError(
            "Папка user_test_dataset не содержит validation-изображений."
        )

    print(f"Исправлений для обучения: {len(correction_samples)}")
    print(f"Validation-изображений пользователя: {len(validation_samples)}")

    mnist_train_data = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    mnist_test_data = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    mnist_test_loader = DataLoader(
        mnist_test_data,
        batch_size=MNIST_EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    candidates = [
        (
            "Базовая CNN",
            BASE_MODEL_FILE,
            torch.load(
                BASE_MODEL_FILE,
                map_location="cpu",
                weights_only=True,
            ),
        )
    ]

    if USER_MODEL_FILE.exists():
        candidates.append(
            (
                "Текущая пользовательская CNN",
                USER_MODEL_FILE,
                torch.load(
                    USER_MODEL_FILE,
                    map_location="cpu",
                    weights_only=True,
                ),
            )
        )

    evaluated_candidates = []

    print()
    print("Оцениваю существующие модели...")

    for name, path, state in candidates:
        (
            mnist_accuracy,
            user_accuracy,
            class_accuracy,
            confusion,
        ) = evaluate_state(
            state,
            mnist_test_loader,
            validation_samples,
        )

        evaluated_candidates.append(
            {
                "name": name,
                "path": path,
                "state": state,
                "mnist_accuracy": mnist_accuracy,
                "user_accuracy": user_accuracy,
                "class_accuracy": class_accuracy,
                "confusion": confusion,
            }
        )

        print(
            f"{name}: MNIST={mnist_accuracy:.2f}% | "
            f"User validation={user_accuracy:.2f}%"
        )

    # Оригинальная CNN всегда остаётся неизменным эталоном безопасности.
    # Поэтому допустимое падение не накапливается между запусками.
    safety_reference = evaluated_candidates[0]
    safety_mnist_accuracy = safety_reference["mnist_accuracy"]
    safety_class_accuracy = safety_reference["class_accuracy"]

    safe_existing_candidates = []

    for candidate in evaluated_candidates:
        candidate_mnist_drop = (
            safety_mnist_accuracy - candidate["mnist_accuracy"]
        )
        candidate_class_drop = maximum_class_drop(
            safety_class_accuracy,
            candidate["class_accuracy"],
        )

        candidate["mnist_drop_from_base"] = candidate_mnist_drop
        candidate["class_drop_from_base"] = candidate_class_drop
        candidate["safe_from_base"] = (
            candidate_mnist_drop <= MAX_MNIST_DROP + 1e-9
            and candidate_class_drop <= MAX_USER_CLASS_DROP + 1e-9
        )

        safety_status = (
            "безопасна"
            if candidate["safe_from_base"]
            else "исключена"
        )

        print(
            f"Проверка {candidate['name']}: "
            f"MNIST drop={candidate_mnist_drop:.2f} | "
            f"class drop={candidate_class_drop:.2f} | "
            f"{safety_status}"
        )

        if candidate["safe_from_base"]:
            safe_existing_candidates.append(candidate)

    # Среди безопасных моделей максимизируем user validation;
    # при равенстве выбираем более сильную модель на MNIST.
    selected_baseline = max(
        safe_existing_candidates,
        key=lambda item: (
            item["user_accuracy"],
            item["mnist_accuracy"],
        ),
    )

    print()
    print("Выбрана исходная модель:", selected_baseline["name"])

    baseline_mnist_accuracy = selected_baseline["mnist_accuracy"]
    baseline_user_accuracy = selected_baseline["user_accuracy"]
    baseline_class_accuracy = selected_baseline["class_accuracy"]

    best_state = {
        name: tensor.clone()
        for name, tensor in selected_baseline["state"].items()
    }

    best_name = selected_baseline["name"]
    best_epoch = 0
    best_mnist_accuracy = baseline_mnist_accuracy
    best_user_accuracy = baseline_user_accuracy
    best_class_accuracy = baseline_class_accuracy

    model = CNN().to(device)
    model.load_state_dict(selected_baseline["state"])

    # Исходная CNN выступает teacher-моделью и никогда не изменяется.
    teacher_model = CNN().to(device)
    teacher_model.load_state_dict(safety_reference["state"])
    teacher_model.eval()

    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    # Копия исходных параметров нужна для L2-SP регуляризации.
    anchor_parameters = {
        name: parameter.detach().clone()
        for name, parameter in teacher_model.named_parameters()
    }

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.features.parameters(),
                "lr": FEATURE_LEARNING_RATE,
            },
            {
                "params": model.classifier.parameters(),
                "lr": CLASSIFIER_LEARNING_RATE,
            },
        ],
    )

    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = STEPS_PER_EPOCH

    history_rows = []
    epochs_without_improvement = 0

    print(f"Шагов в эпохе: {steps_per_epoch}")
    print()

    for epoch in range(1, EPOCHS + 1):
        train_losses = train_one_epoch(
            model,
            teacher_model,
            optimizer,
            loss_fn,
            correction_samples,
            mnist_train_data,
            steps_per_epoch,
            anchor_parameters,
        )

        mnist_accuracy = evaluate_mnist(
            model,
            mnist_test_loader,
        )

        (
            user_accuracy,
            class_accuracy,
            _,
        ) = evaluate_user_validation(
            model,
            validation_samples,
        )

        mnist_drop = safety_mnist_accuracy - mnist_accuracy
        class_drop = maximum_class_drop(
            safety_class_accuracy,
            class_accuracy,
        )

        mnist_safe = (
            mnist_drop <= MAX_MNIST_DROP + 1e-9
        )

        class_safe = (
            class_drop <= MAX_USER_CLASS_DROP + 1e-9
        )

        safe = mnist_safe and class_safe

        better = is_better_model(
            user_accuracy,
            mnist_accuracy,
            best_user_accuracy,
            best_mnist_accuracy,
        )

        accepted = safe and better

        if accepted:
            best_state = copy_state_dict_to_cpu(model)
            best_name = f"Безопасная модель эпохи {epoch}"
            best_epoch = epoch
            best_mnist_accuracy = mnist_accuracy
            best_user_accuracy = user_accuracy
            best_class_accuracy = class_accuracy
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                **train_losses,
                "mnist_accuracy": mnist_accuracy,
                "user_validation_accuracy": user_accuracy,
                "mnist_drop": mnist_drop,
                "max_user_class_drop": class_drop,
                "mnist_safe": mnist_safe,
                "class_safe": class_safe,
                "safe": safe,
                "better_than_best": better,
                "accepted_as_best": accepted,
            }
        )

        status = "ACCEPTED" if accepted else "rejected"

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={train_losses['total_loss']:.4f} | "
            f"MNIST={mnist_accuracy:.2f}% | "
            f"User val={user_accuracy:.2f}% | "
            f"MNIST drop={mnist_drop:.2f} | "
            f"max class drop={class_drop:.2f} | "
            f"{status}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Ранняя остановка: "
                f"{EARLY_STOPPING_PATIENCE} эпохи подряд "
                "без нового безопасного улучшения."
            )
            break

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_path = None

    if USER_MODEL_FILE.exists():
        backup_path = VERSIONS_DIR / (
            f"mnist_cnn_user_before_{timestamp}.pth"
        )

        shutil.copy2(
            USER_MODEL_FILE,
            backup_path,
        )

    selected_version_path = VERSIONS_DIR / (
        f"mnist_cnn_selected_{timestamp}.pth"
    )

    torch.save(best_state, selected_version_path)
    torch.save(best_state, USER_MODEL_FILE)

    with HISTORY_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "total_loss",
                "classification_loss",
                "distillation_loss",
                "anchor_loss",
                "mnist_accuracy",
                "user_validation_accuracy",
                "mnist_drop",
                "max_user_class_drop",
                "mnist_safe",
                "class_safe",
                "safe",
                "better_than_best",
                "accepted_as_best",
            ],
        )

        writer.writeheader()
        writer.writerows(history_rows)

    report_lines = [
        "БЕЗОПАСНОЕ ПАКЕТНОЕ ДООБУЧЕНИЕ CNN",
        "=" * 72,
        f"Устройство: {device}",
        f"Исправлений для обучения: {len(correction_samples)}",
        f"Validation-изображений: {len(validation_samples)}",
        f"Фактически выполнено эпох: {len(history_rows)}",
        f"Пользовательских примеров в batch: {USER_SAMPLES_PER_BATCH}",
        f"MNIST replay-примеров в batch: {BATCH_SIZE - USER_SAMPLES_PER_BATCH}",
        f"Шагов в эпохе: {STEPS_PER_EPOCH}",
        f"Learning rate свёрточных слоёв: {FEATURE_LEARNING_RATE}",
        f"Learning rate классификатора: {CLASSIFIER_LEARNING_RATE}",
        f"Вес distillation: {DISTILLATION_WEIGHT}",
        f"Сила привязки весов: {WEIGHT_ANCHOR_STRENGTH}",
        f"Допустимое падение MNIST: {MAX_MNIST_DROP:.2f} пункта",
        (
            "Допустимое падение отдельной пользовательской цифры: "
            f"{MAX_USER_CLASS_DROP:.2f} пункта"
        ),
        "",
        f"Эталон безопасности: {safety_reference['name']}",
        f"Эталонная MNIST accuracy: {safety_mnist_accuracy:.2f}%",
        "",
        f"Исходная выбранная модель: {selected_baseline['name']}",
        f"Исходная MNIST accuracy: {baseline_mnist_accuracy:.2f}%",
        f"Исходная User validation accuracy: {baseline_user_accuracy:.2f}%",
        "Исходная точность по цифрам:",
        format_class_accuracy(baseline_class_accuracy),
        "",
        f"Итоговая выбранная модель: {best_name}",
        f"Выбранная эпоха: {best_epoch}",
        f"Итоговая MNIST accuracy: {best_mnist_accuracy:.2f}%",
        f"Итоговая User validation accuracy: {best_user_accuracy:.2f}%",
        "Итоговая точность по цифрам:",
        format_class_accuracy(best_class_accuracy),
        "",
        f"Рабочая модель сохранена: {USER_MODEL_FILE.resolve()}",
        f"Версия модели сохранена: {selected_version_path.resolve()}",
        (
            f"Резервная копия прежней модели: {backup_path.resolve()}"
            if backup_path is not None
            else "Прежней пользовательской модели не было."
        ),
        f"История эпох: {HISTORY_FILE.resolve()}",
        "",
        (
            "Текущий user_test_dataset теперь является validation-набором. "
            "После завершения настройки нужно собрать новый независимый "
            "user_final_test_dataset."
        ),
    ]

    report_text = "\n".join(report_lines)
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print()
    print(report_text)
    print()
    print(f"Отчёт: {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()
