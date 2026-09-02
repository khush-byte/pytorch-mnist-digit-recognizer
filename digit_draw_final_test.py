import os
import random
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Устройство:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# Базовые веса получены после обучения mnist_cnn_train.py.
# Пользовательская версия создаётся после исправлений и имеет
# отдельное имя, поэтому она не конфликтует со старой Linear-моделью.
BASE_MODEL_FILE = "mnist_cnn_model.pth"
USER_MODEL_FILE = "mnist_cnn_user.pth"

CORRECTIONS_DIR = "corrections"
TEST_DATASET_DIR = "user_final_test_dataset"
TEST_REPORT_FILE = "user_final_test_evaluation.txt"
TEST_ERRORS_IMAGE_FILE = "user_final_test_errors.png"
TARGET_TEST_SAMPLES_PER_DIGIT = 20

# В этой специальной версии веса модели зафиксированы. Финальные
# изображения нельзя использовать для коррекции или дообучения.
FINAL_TEST_MODE = True

os.makedirs(
    CORRECTIONS_DIR,
    exist_ok=True
)

os.makedirs(
    TEST_DATASET_DIR,
    exist_ok=True
)


# ============================================================
# 2. АРХИТЕКТУРА МОДЕЛИ
#
# ВАЖНО: архитектура полностью совпадает с mnist_cnn_train.py,
# на котором был создан файл mnist_cnn_model.pth.
# ============================================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = CNN().to(device)


# ============================================================
# 3. ЗАГРУЗКА ВЕСОВ
# ============================================================

def load_model_weights():

    global model

    if not os.path.exists(BASE_MODEL_FILE):
        raise FileNotFoundError(
            f"Не найден файл {BASE_MODEL_FILE}.\n"
            "Поместите программу в ту же папку, где находится обученная модель."
        )

    # Если уже есть модель, обученная на исправлениях пользователя,
    # сначала пытаемся загрузить её.
    if os.path.exists(USER_MODEL_FILE):

        try:
            state = torch.load(
                USER_MODEL_FILE,
                map_location=device,
                weights_only=True
            )

            model.load_state_dict(state)

            model.eval()

            print(
                "Загружена модель с пользовательскими исправлениями:",
                USER_MODEL_FILE
            )

            return USER_MODEL_FILE

        except Exception as error:

            print(
                "Не удалось загрузить пользовательскую модель:",
                error
            )

            print(
                "Возвращаюсь к базовой модели."
            )

    state = torch.load(
        BASE_MODEL_FILE,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)

    model.eval()

    print(
        "Загружена базовая модель:",
        BASE_MODEL_FILE
    )

    return BASE_MODEL_FILE


current_model_file = load_model_weights()


# ============================================================
# 4. MNIST ДЛЯ REPLAY ПРИ ДООБУЧЕНИИ
#
# Мы не обучаем модель только на одной ошибке.
# При коррекции смешиваем пользовательские рисунки с обычными
# примерами MNIST, чтобы уменьшить риск "сломать" старые знания.
# ============================================================

replay_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


# Небольшая аугментация пользовательских исправлений.
# Это помогает модели не просто запомнить одну конкретную картинку.
correction_augmentation = transforms.Compose([
    transforms.RandomAffine(
        degrees=8,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        shear=4,
        fill=0
    ),
    transforms.ToTensor()
])


# ============================================================
# 5. ОСНОВНОЕ ОКНО
# ============================================================

root = tk.Tk()

root.title(
    "Финальный независимый тест CNN на рукописных цифрах"
)

# Разворачиваем окно на весь экран Windows,
# но оставляем панель задач и стандартные кнопки окна.
root.state("zoomed")
root.resizable(True, True)

root.update_idletasks()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Размер поля подстраивается под экран.
CANVAS_SIZE = max(
    320,
    min(
        560,
        screen_height - 300,
        screen_width // 2 - 80
    )
)

# Тонкая кисть.
BRUSH_SIZE = 14


# ============================================================
# 6. ПЕРЕМЕННЫЕ СОСТОЯНИЯ
# ============================================================

image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    color=0
)

draw = ImageDraw.Draw(image)

last_x = None
last_y = None

# Последняя картинка 28x28, которую реально увидела модель.
last_mnist_image = None

# Последнее предсказание модели.
last_prediction = None


# ============================================================
# 7. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def count_corrections():

    total = 0

    for digit in range(10):

        directory = os.path.join(
            CORRECTIONS_DIR,
            str(digit)
        )

        if not os.path.isdir(directory):
            continue

        total += sum(
            1
            for name in os.listdir(directory)
            if name.lower().endswith(".png")
        )

    return total


def get_test_sample_counts():

    counts = {}

    for digit in range(10):

        directory = os.path.join(
            TEST_DATASET_DIR,
            str(digit)
        )

        if not os.path.isdir(directory):
            counts[digit] = 0
            continue

        counts[digit] = sum(
            1
            for name in os.listdir(directory)
            if name.lower().endswith(".png")
        )

    return counts


def update_model_info():

    corrections = count_corrections()

    model_name = current_model_file

    model_info_label.config(
        text=(
            f"Модель: {model_name}   |   "
            f"Исправлений сохранено: {corrections}   |   "
            f"Устройство: {device}"
        )
    )


def update_test_info():

    counts = get_test_sample_counts()

    text = "   ".join(
        f"{digit}: {counts[digit]}/{TARGET_TEST_SAMPLES_PER_DIGIT}"
        for digit in range(10)
    )

    test_counts_label.config(
        text=text
    )


def set_training_controls(enabled):

    state = tk.NORMAL if enabled else tk.DISABLED

    # Предсказание и изменение весов отключены, чтобы пользователь
    # не отбирал изображения по ответу модели и не загрязнял final test.
    locked_state = (
        tk.DISABLED
        if FINAL_TEST_MODE
        else state
    )

    predict_button.config(state=locked_state)
    clear_button.config(state=state)
    correction_button.config(state=locked_state)
    train_all_button.config(state=locked_state)
    reset_model_button.config(state=locked_state)
    correct_digit_entry.config(state=locked_state)
    save_test_button.config(state=state)
    evaluate_test_button.config(state=state)
    open_errors_button.config(state=state)
    test_digit_entry.config(state=state)


def format_probabilities(probabilities):

    lines = []

    for start in (0, 5):

        parts = []

        for digit in range(start, start + 5):

            probability = (
                probabilities[0, digit].item()
                * 100
            )

            parts.append(
                f"{digit}: {probability:6.2f}%"
            )

        lines.append(
            "     ".join(parts)
        )

    return "\n".join(lines)


def show_prediction(probabilities):

    global last_prediction

    predicted_digit = probabilities.argmax(
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_digit
    ].item() * 100

    last_prediction = predicted_digit

    result_label.config(
        text=f"AI считает, что это: {predicted_digit}"
    )

    confidence_label.config(
        text=f"Уверенность: {confidence:.2f}%"
    )

    probability_label.config(
        text=format_probabilities(
            probabilities
        )
    )

    return predicted_digit, confidence


# ============================================================
# 8. РИСОВАНИЕ
# ============================================================

def start_draw(event):

    global last_x, last_y

    last_x = event.x
    last_y = event.y

    radius = BRUSH_SIZE // 2

    canvas.create_oval(
        event.x - radius,
        event.y - radius,
        event.x + radius,
        event.y + radius,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius
        ),
        fill=255
    )


def draw_digit(event):

    global last_x, last_y

    if last_x is None or last_y is None:
        return

    canvas.create_line(
        last_x,
        last_y,
        event.x,
        event.y,
        fill="white",
        width=BRUSH_SIZE,
        capstyle=tk.ROUND,
        smooth=True
    )

    draw.line(
        (
            last_x,
            last_y,
            event.x,
            event.y
        ),
        fill=255,
        width=BRUSH_SIZE
    )

    radius = BRUSH_SIZE // 2

    draw.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius
        ),
        fill=255
    )

    last_x = event.x
    last_y = event.y


def stop_draw(event):

    global last_x, last_y

    last_x = None
    last_y = None


# ============================================================
# 9. ПОДГОТОВКА РИСУНКА К MNIST
# ============================================================

def prepare_image():

    bbox = image.getbbox()

    if bbox is None:
        return None

    # Обрезаем пустое пространство.
    cropped = image.crop(bbox)

    width, height = cropped.size

    # В MNIST сама цифра обычно занимает примерно область 20x20
    # внутри изображения 28x28.
    max_size = 20

    scale = max_size / max(
        width,
        height
    )

    new_width = max(
        1,
        round(width * scale)
    )

    new_height = max(
        1,
        round(height * scale)
    )

    cropped = cropped.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )

    mnist_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # Первичное геометрическое центрирование.
    x = (
        28 - new_width
    ) // 2

    y = (
        28 - new_height
    ) // 2

    mnist_image.paste(
        cropped,
        (x, y)
    )

    # --------------------------------------------------------
    # Центрирование по центру массы пикселей.
    # Это делает пользовательский рисунок ближе к MNIST.
    # --------------------------------------------------------

    pixels = mnist_image.load()

    total = 0
    sum_x = 0
    sum_y = 0

    for py in range(28):

        for px in range(28):

            value = pixels[
                px,
                py
            ]

            total += value
            sum_x += px * value
            sum_y += py * value

    if total > 0:

        center_x = (
            sum_x / total
        )

        center_y = (
            sum_y / total
        )

        shift_x = round(
            13.5 - center_x
        )

        shift_y = round(
            13.5 - center_y
        )

        centered = Image.new(
            "L",
            (28, 28),
            0
        )

        centered.paste(
            mnist_image,
            (
                shift_x,
                shift_y
            )
        )

        mnist_image = centered

    # Для диагностики:
    # это именно та картинка, которую получает нейросеть.
    mnist_image.save(
        "last_input.png"
    )

    return mnist_image


# ============================================================
# 10. РАСПОЗНАВАНИЕ
# ============================================================

def predict():

    global last_mnist_image

    mnist_image = prepare_image()

    if mnist_image is None:

        result_label.config(
            text="Сначала нарисуйте цифру"
        )

        confidence_label.config(
            text=""
        )

        probability_label.config(
            text=""
        )

        correction_status.config(
            text=""
        )

        return

    last_mnist_image = (
        mnist_image.copy()
    )

    tensor = ToTensor()(
        mnist_image
    )

    # [1, 28, 28] -> [1, 1, 28, 28]
    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    predicted_digit, confidence = (
        show_prediction(
            probabilities
        )
    )

    correction_status.config(
        text=(
            f"Получен ответ {predicted_digit}. "
            "Если он неверный — укажите правильную цифру справа."
        )
    )


# ============================================================
# 11. СОХРАНЕНИЕ ПОЛЬЗОВАТЕЛЬСКОГО ИСПРАВЛЕНИЯ
# ============================================================

def save_correction(
    mnist_image,
    correct_digit
):

    label_directory = os.path.join(
        CORRECTIONS_DIR,
        str(correct_digit)
    )

    os.makedirs(
        label_directory,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    path = os.path.join(
        label_directory,
        f"{timestamp}.png"
    )

    mnist_image.save(
        path
    )

    return path


# ============================================================
# 12. СОЗДАНИЕ REPLAY-BATCH
# ============================================================

def make_replay_batch(
    correction_images,
    correction_labels,
    replay_count
):

    replay_indexes = random.sample(
        range(
            len(replay_dataset)
        ),
        replay_count
    )

    replay_images = []
    replay_labels = []

    for index in replay_indexes:

        img, label = replay_dataset[
            index
        ]

        replay_images.append(
            img
        )

        replay_labels.append(
            label
        )

    replay_images = torch.stack(
        replay_images
    )

    replay_labels = torch.tensor(
        replay_labels,
        dtype=torch.long
    )

    user_images = torch.stack(
        correction_images
    )

    user_labels = torch.tensor(
        correction_labels,
        dtype=torch.long
    )

    images_batch = torch.cat(
        [
            user_images,
            replay_images
        ],
        dim=0
    )

    labels_batch = torch.cat(
        [
            user_labels,
            replay_labels
        ],
        dim=0
    )

    # Перемешиваем batch.
    permutation = torch.randperm(
        labels_batch.size(0)
    )

    images_batch = images_batch[
        permutation
    ]

    labels_batch = labels_batch[
        permutation
    ]

    return (
        images_batch.to(device),
        labels_batch.to(device)
    )


# ============================================================
# 13. ИСПРАВИТЬ ОШИБКУ И НЕМНОГО ДООБУЧИТЬ МОДЕЛЬ
# ============================================================

def correct_and_train():

    global last_prediction
    global current_model_file

    if FINAL_TEST_MODE:
        correction_status.config(
            text=(
                "Финальный тест активен: коррекция и изменение "
                "весов модели заблокированы."
            )
        )
        return

    if last_mnist_image is None:

        correction_status.config(
            text=(
                "Сначала нарисуйте цифру "
                "и нажмите «Распознать»."
            )
        )

        return

    value = (
        correct_digit_var
        .get()
        .strip()
    )

    if (
        len(value) != 1
        or not value.isdigit()
    ):

        correction_status.config(
            text="Введите одну правильную цифру от 0 до 9."
        )

        return

    correct_digit = int(
        value
    )

    if (
        last_prediction
        == correct_digit
    ):

        correction_status.config(
            text=(
                "Модель уже дала этот ответ. "
                "Исправление не требуется."
            )
        )

        return

    path = save_correction(
        last_mnist_image,
        correct_digit
    )

    set_training_controls(
        False
    )

    correction_status.config(
        text=(
            "Исправление сохранено. "
            "Дообучаю модель..."
        )
    )

    root.update_idletasks()

    try:

        model.train()

        loss_fn = nn.CrossEntropyLoss()

        # Маленький learning rate:
        # меняем модель аккуратно.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001
        )

        training_steps = 30

        # Из batch 64:
        # 8 пользовательских примеров + 56 MNIST.
        correction_count = 8
        replay_count = 56

        final_loss = 0.0

        for step in range(
            training_steps
        ):

            correction_images = []
            correction_labels = []

            # Один оригинал.
            correction_images.append(
                ToTensor()(
                    last_mnist_image
                )
            )

            correction_labels.append(
                correct_digit
            )

            # Остальные — небольшие вариации рисунка.
            for _ in range(
                correction_count - 1
            ):

                augmented = (
                    correction_augmentation(
                        last_mnist_image
                    )
                )

                correction_images.append(
                    augmented
                )

                correction_labels.append(
                    correct_digit
                )

            (
                images_batch,
                labels_batch
            ) = make_replay_batch(
                correction_images,
                correction_labels,
                replay_count
            )

            logits = model(
                images_batch
            )

            loss = loss_fn(
                logits,
                labels_batch
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            final_loss = (
                loss.item()
            )

        model.eval()

        torch.save(
            model.state_dict(),
            USER_MODEL_FILE
        )

        current_model_file = (
            USER_MODEL_FILE
        )

        # Проверяем тот же пример после обучения.
        test_tensor = (
            ToTensor()(
                last_mnist_image
            )
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            logits = model(
                test_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        new_prediction, new_confidence = (
            show_prediction(
                probabilities
            )
        )

        update_model_info()

        if (
            new_prediction
            == correct_digit
        ):

            correction_status.config(
                text=(
                    f"✓ Исправление принято. "
                    f"Теперь модель отвечает: {new_prediction} "
                    f"({new_confidence:.2f}%). "
                    f"Loss: {final_loss:.4f}"
                )
            )

        else:

            correction_status.config(
                text=(
                    f"Исправление сохранено, но после короткого "
                    f"дообучения модель пока отвечает: "
                    f"{new_prediction} ({new_confidence:.2f}%). "
                    "Добавьте ещё похожие примеры или нажмите "
                    "«Переобучить на всех исправлениях»."
                )
            )

        print()
        print(
            "Исправление сохранено:",
            path
        )
        print(
            "Правильный ответ:",
            correct_digit
        )
        print(
            "Loss:",
            final_loss
        )
        print(
            "Обновлённая модель:",
            USER_MODEL_FILE
        )

    except Exception as error:

        correction_status.config(
            text=(
                "Ошибка при дообучении: "
                f"{error}"
            )
        )

        print(
            "Ошибка дообучения:",
            error
        )

    finally:

        set_training_controls(
            True
        )


# ============================================================
# 14. ЗАГРУЗКА ВСЕХ СОХРАНЁННЫХ ИСПРАВЛЕНИЙ
# ============================================================

def load_all_corrections():

    samples = []

    for digit in range(10):

        directory = os.path.join(
            CORRECTIONS_DIR,
            str(digit)
        )

        if not os.path.isdir(
            directory
        ):
            continue

        for filename in os.listdir(
            directory
        ):

            if not filename.lower().endswith(
                ".png"
            ):
                continue

            path = os.path.join(
                directory,
                filename
            )

            samples.append(
                (
                    path,
                    digit
                )
            )

    return samples


# ============================================================
# 15. ПЕРЕОБУЧЕНИЕ НА ВСЕХ НАКОПЛЕННЫХ ИСПРАВЛЕНИЯХ
# ============================================================

def train_on_all_corrections():

    global current_model_file

    if FINAL_TEST_MODE:
        correction_status.config(
            text=(
                "Финальный тест активен: пакетное дообучение "
                "заблокировано."
            )
        )
        return

    samples = (
        load_all_corrections()
    )

    if not samples:

        correction_status.config(
            text=(
                "Пока нет сохранённых исправлений."
            )
        )

        return

    set_training_controls(
        False
    )

    correction_status.config(
        text=(
            f"Переобучение на {len(samples)} "
            "сохранённых исправлениях..."
        )
    )

    root.update_idletasks()

    try:

        model.train()

        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.00008
        )

        # Чем больше накоплено исправлений,
        # тем больше шагов, но ограничиваем время.
        training_steps = max(
            50,
            min(
                400,
                len(samples) * 6
            )
        )

        user_per_batch = min(
            16,
            max(
                4,
                len(samples)
            )
        )

        replay_count = (
            64 - user_per_batch
        )

        final_loss = 0.0

        for step in range(
            training_steps
        ):

            selected = random.choices(
                samples,
                k=user_per_batch
            )

            correction_images = []
            correction_labels = []

            for path, digit in selected:

                with Image.open(
                    path
                ) as opened:

                    user_image = (
                        opened
                        .convert("L")
                        .copy()
                    )

                # В половине случаев используем оригинал,
                # в остальных — слегка изменённый вариант.
                if random.random() < 0.5:

                    tensor = ToTensor()(
                        user_image
                    )

                else:

                    tensor = (
                        correction_augmentation(
                            user_image
                        )
                    )

                correction_images.append(
                    tensor
                )

                correction_labels.append(
                    digit
                )

            (
                images_batch,
                labels_batch
            ) = make_replay_batch(
                correction_images,
                correction_labels,
                replay_count
            )

            logits = model(
                images_batch
            )

            loss = loss_fn(
                logits,
                labels_batch
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            final_loss = (
                loss.item()
            )

            if (
                step % 25 == 0
                or step
                == training_steps - 1
            ):

                correction_status.config(
                    text=(
                        f"Переобучение: "
                        f"{step + 1}/{training_steps} "
                        f"| Loss: {final_loss:.4f}"
                    )
                )

                root.update_idletasks()

        model.eval()

        torch.save(
            model.state_dict(),
            USER_MODEL_FILE
        )

        current_model_file = (
            USER_MODEL_FILE
        )

        update_model_info()

        correction_status.config(
            text=(
                f"✓ Переобучение завершено. "
                f"Использовано исправлений: {len(samples)}. "
                f"Финальный loss: {final_loss:.4f}"
            )
        )

    except Exception as error:

        correction_status.config(
            text=(
                "Ошибка при переобучении: "
                f"{error}"
            )
        )

        print(
            "Ошибка переобучения:",
            error
        )

    finally:

        model.eval()

        set_training_controls(
            True
        )


# ============================================================
# 16. СБРОС ПОЛЬЗОВАТЕЛЬСКОЙ МОДЕЛИ
#
# Сохранённые картинки-исправления НЕ удаляются.
# Удаляется только mnist_cnn_user.pth.
# ============================================================

def reset_user_model():

    global current_model_file
    global last_prediction

    if FINAL_TEST_MODE:
        correction_status.config(
            text=(
                "Финальный тест активен: смена модели заблокирована."
            )
        )
        return

    answer = messagebox.askyesno(
        "Сброс модели",
        (
            "Вернуть исходную CNN-модель mnist_cnn_model.pth?\n\n"
            "Сохранённые рисунки в папке corrections "
            "останутся и их можно будет использовать снова."
        )
    )

    if not answer:
        return

    try:

        if os.path.exists(
            USER_MODEL_FILE
        ):

            os.remove(
                USER_MODEL_FILE
            )

        state = torch.load(
            BASE_MODEL_FILE,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(
            state
        )

        model.eval()

        current_model_file = (
            BASE_MODEL_FILE
        )

        last_prediction = None

        update_model_info()

        correction_status.config(
            text=(
                "Модель сброшена к исходной версии."
            )
        )

    except Exception as error:

        correction_status.config(
            text=(
                "Не удалось сбросить модель: "
                f"{error}"
            )
        )


# ============================================================
# 17. НЕЗАВИСИМЫЙ ТЕСТОВЫЙ НАБОР ПОЛЬЗОВАТЕЛЯ
#
# Эти изображения никогда не используются для обучения.
# Они нужны только для честной проверки модели на реальном почерке.
# ============================================================

def save_test_sample():

    value = (
        test_digit_var
        .get()
        .strip()
    )

    if (
        len(value) != 1
        or not value.isdigit()
    ):

        test_status.config(
            text="Укажите правильную цифру от 0 до 9."
        )

        return

    digit = int(value)

    counts_before_save = get_test_sample_counts()

    if counts_before_save[digit] >= TARGET_TEST_SAMPLES_PER_DIGIT:

        test_status.config(
            text=(
                f"Для цифры {digit} уже собрано "
                f"{TARGET_TEST_SAMPLES_PER_DIGIT} изображений. "
                "Выберите следующую цифру."
            )
        )

        return

    mnist_image = prepare_image()

    if mnist_image is None:

        test_status.config(
            text="Сначала нарисуйте цифру."
        )

        return

    directory = os.path.join(
        TEST_DATASET_DIR,
        str(digit)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    path = os.path.join(
        directory,
        f"{timestamp}.png"
    )

    mnist_image.save(path)

    counts = get_test_sample_counts()
    saved_count = counts[digit]

    # Очищаем поле, но сохраняем выбранную тестовую метку.
    clear()
    test_digit_var.set(str(digit))
    update_test_info()

    test_status.config(
        text=(
            f"✓ Цифра {digit} сохранена только для final test: "
            f"{saved_count}/{TARGET_TEST_SAMPLES_PER_DIGIT}. "
            "Нарисуйте следующий вариант."
        )
    )

    print(
        "Финальный тестовый пример сохранён:",
        path
    )


def load_user_test_samples():

    samples = []

    for digit in range(10):

        directory = os.path.join(
            TEST_DATASET_DIR,
            str(digit)
        )

        if not os.path.isdir(directory):
            continue

        for filename in sorted(
            os.listdir(directory)
        ):

            if not filename.lower().endswith(".png"):
                continue

            samples.append(
                (
                    os.path.join(directory, filename),
                    digit
                )
            )

    return samples


def evaluate_model_on_user_samples(
    model_file,
    samples
):

    evaluation_model = CNN().to(device)

    state = torch.load(
        model_file,
        map_location=device,
        weights_only=True
    )

    evaluation_model.load_state_dict(state)
    evaluation_model.eval()

    confusion = torch.zeros(
        (10, 10),
        dtype=torch.int64
    )

    error_details = []

    correct = 0
    total = 0
    batch_size = 128

    with torch.inference_mode():

        for start in range(
            0,
            len(samples),
            batch_size
        ):

            batch_samples = samples[
                start:start + batch_size
            ]

            images = []
            labels = []

            for path, digit in batch_samples:

                with Image.open(path) as opened:

                    sample_image = (
                        opened
                        .convert("L")
                        .copy()
                    )

                images.append(
                    ToTensor()(sample_image)
                )

                labels.append(digit)

            images_batch = torch.stack(images).to(device)

            labels_cpu = torch.tensor(
                labels,
                dtype=torch.long
            )

            labels_device = labels_cpu.to(device)

            logits = evaluation_model(images_batch)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = probabilities.argmax(dim=1)

            predicted_confidences = probabilities.gather(
                1,
                predictions.unsqueeze(1)
            ).squeeze(1)

            correct += (
                predictions == labels_device
            ).sum().item()

            total += len(labels)

            pairs = (
                labels_cpu * 10
                + predictions.cpu()
            )

            confusion += torch.bincount(
                pairs,
                minlength=100
            ).reshape(10, 10)

            for index, (path, correct_digit) in enumerate(
                batch_samples
            ):

                predicted_digit = predictions[index].item()

                if predicted_digit == correct_digit:
                    continue

                error_details.append(
                    {
                        "path": path,
                        "correct_digit": correct_digit,
                        "predicted_digit": predicted_digit,
                        "confidence": (
                            predicted_confidences[index].item()
                            * 100
                        )
                    }
                )

    accuracy = (
        100.0 * correct / total
        if total > 0
        else 0.0
    )

    del evaluation_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return accuracy, confusion, error_details


def build_user_test_report_section(
    model_name,
    model_file,
    accuracy,
    confusion,
    error_details
):

    lines = [
        "=" * 72,
        f"Модель: {model_name}",
        f"Файл: {model_file}",
        f"Общая точность: {accuracy:.2f}%",
        "",
        "Точность по каждой цифре:"
    ]

    for digit in range(10):

        total = confusion[digit].sum().item()
        correct = confusion[digit, digit].item()

        if total == 0:

            lines.append(
                f"  {digit}: нет тестовых примеров"
            )

        else:

            class_accuracy = (
                100.0 * correct / total
            )

            lines.append(
                f"  {digit}: {class_accuracy:6.2f}% "
                f"({correct}/{total})"
            )

    lines.extend(
        [
            "",
            "Матрица ошибок:",
            "Строка = правильная цифра, столбец = ответ модели.",
            "",
            "прав.\\ответ"
            + "".join(
                f"{digit:>6}"
                for digit in range(10)
            )
        ]
    )

    for digit in range(10):

        row = "".join(
            f"{confusion[digit, predicted].item():>6}"
            for predicted in range(10)
        )

        lines.append(
            f"{digit:>11}{row}"
        )

    errors = []

    for correct_digit in range(10):

        for predicted_digit in range(10):

            if correct_digit == predicted_digit:
                continue

            count = confusion[
                correct_digit,
                predicted_digit
            ].item()

            if count > 0:
                errors.append(
                    (
                        count,
                        correct_digit,
                        predicted_digit
                    )
                )

    errors.sort(reverse=True)

    lines.extend(
        [
            "",
            "Самые частые ошибки:"
        ]
    )

    if not errors:

        lines.append(
            "  Ошибок нет."
        )

    else:

        for count, correct_digit, predicted_digit in errors[:10]:

            lines.append(
                f"  цифра {correct_digit} распознана как "
                f"{predicted_digit}: {count} раз"
            )

    lines.extend(
        [
            "",
            "Подробности по каждому ошибочному изображению:"
        ]
    )

    if not error_details:

        lines.append(
            "  Ошибочных изображений нет."
        )

    else:

        sorted_details = sorted(
            error_details,
            key=lambda item: item["confidence"],
            reverse=True
        )

        for number, error in enumerate(
            sorted_details,
            start=1
        ):

            lines.append(
                f"  {number}. Правильная цифра: "
                f"{error['correct_digit']} | "
                f"Ответ модели: {error['predicted_digit']} | "
                f"Уверенность: {error['confidence']:.2f}%"
            )

            lines.append(
                f"     Файл: {error['path']}"
            )

    return lines


def get_gallery_font(size):

    font_candidates = [
        "arial.ttf",
        "DejaVuSans.ttf"
    ]

    for font_name in font_candidates:

        try:
            return ImageFont.truetype(
                font_name,
                size
            )

        except OSError:
            continue

    return ImageFont.load_default()


def create_error_gallery(all_error_details):

    columns = 3
    card_width = 290
    card_height = 180
    margin = 20
    header_height = 65

    item_count = max(
        1,
        len(all_error_details)
    )

    rows = (
        item_count + columns - 1
    ) // columns

    gallery_width = (
        margin * 2
        + columns * card_width
    )

    gallery_height = (
        header_height
        + margin
        + rows * card_height
        + margin
    )

    gallery = Image.new(
        "RGB",
        (
            gallery_width,
            gallery_height
        ),
        "#f2f2f2"
    )

    gallery_draw = ImageDraw.Draw(gallery)

    title_font = get_gallery_font(24)
    text_font = get_gallery_font(16)
    small_font = get_gallery_font(13)

    gallery_draw.text(
        (margin, 18),
        "CNN handwriting error analysis",
        fill="black",
        font=title_font
    )

    if not all_error_details:

        gallery_draw.text(
            (
                margin,
                header_height + 35
            ),
            "No errors: every test image was classified correctly.",
            fill="#176b2c",
            font=text_font
        )

        gallery.save(
            TEST_ERRORS_IMAGE_FILE
        )

        return 0

    for index, error in enumerate(
        all_error_details
    ):

        row = index // columns
        column = index % columns

        x = margin + column * card_width
        y = header_height + margin + row * card_height

        gallery_draw.rounded_rectangle(
            (
                x,
                y,
                x + card_width - 12,
                y + card_height - 12
            ),
            radius=10,
            fill="white",
            outline="#bbbbbb",
            width=2
        )

        with Image.open(
            error["path"]
        ) as opened:

            digit_image = (
                opened
                .convert("RGB")
                .resize(
                    (112, 112),
                    Image.Resampling.NEAREST
                )
            )

        gallery.paste(
            digit_image,
            (
                x + 12,
                y + 18
            )
        )

        model_label = (
            "Base CNN"
            if error["model_name"] == "Базовая CNN"
            else "User CNN"
        )

        text_x = x + 137

        gallery_draw.text(
            (text_x, y + 18),
            model_label,
            fill="#1f4e79",
            font=text_font
        )

        gallery_draw.text(
            (text_x, y + 52),
            f"True: {error['correct_digit']}",
            fill="black",
            font=text_font
        )

        gallery_draw.text(
            (text_x, y + 78),
            f"Predicted: {error['predicted_digit']}",
            fill="#a80000",
            font=text_font
        )

        gallery_draw.text(
            (text_x, y + 104),
            f"Confidence: {error['confidence']:.2f}%",
            fill="black",
            font=small_font
        )

        filename = os.path.basename(
            error["path"]
        )

        gallery_draw.text(
            (x + 12, y + 140),
            filename,
            fill="#555555",
            font=small_font
        )

    gallery.save(
        TEST_ERRORS_IMAGE_FILE
    )

    return len(all_error_details)


def open_errors_image():

    if not os.path.exists(
        TEST_ERRORS_IMAGE_FILE
    ):

        test_status.config(
            text=(
                "Изображение ошибок ещё не создано. "
                "Сначала запустите проверку моделей."
            )
        )

        return

    image_path = os.path.abspath(
        TEST_ERRORS_IMAGE_FILE
    )

    try:

        if hasattr(os, "startfile"):
            os.startfile(image_path)
        else:
            messagebox.showinfo(
                "Изображение ошибок",
                image_path
            )

    except Exception as error:

        test_status.config(
            text=(
                "Не удалось открыть изображение ошибок: "
                f"{error}"
            )
        )


def evaluate_user_test_dataset():

    samples = load_user_test_samples()

    if not samples:

        test_status.config(
            text=(
                "Final test пуст. Сначала сохраните "
                "новые рисунки."
            )
        )

        return

    counts = get_test_sample_counts()
    incomplete_digits = [
        f"{digit}: {counts[digit]}/{TARGET_TEST_SAMPLES_PER_DIGIT}"
        for digit in range(10)
        if counts[digit] < TARGET_TEST_SAMPLES_PER_DIGIT
    ]

    if incomplete_digits:

        messagebox.showwarning(
            "Final test ещё не готов",
            (
                "Для честной сбалансированной оценки нужно по "
                f"{TARGET_TEST_SAMPLES_PER_DIGIT} новых изображений "
                "каждой цифры.\n\nНе завершены:\n"
                + "\n".join(incomplete_digits)
            ),
        )
        return

    set_training_controls(False)

    test_status.config(
        text=(
            f"Выполняю финальную оценку моделей на {len(samples)} "
            "новых рисунках вашего почерка..."
        )
    )

    root.update_idletasks()

    try:

        models_to_test = [
            (
                "Базовая CNN",
                BASE_MODEL_FILE
            )
        ]

        if os.path.exists(USER_MODEL_FILE):

            models_to_test.append(
                (
                    "CNN после пользовательских исправлений",
                    USER_MODEL_FILE
                )
            )

        report = [
            "ФИНАЛЬНАЯ НЕЗАВИСИМАЯ ОЦЕНКА CNN НА ПОЧЕРКЕ ПОЛЬЗОВАТЕЛЯ",
            f"Количество изображений: {len(samples)}",
            f"Устройство: {device}",
            "",
            (
                "Важно: эти изображения не использовались "
                "ни для обучения, ни для выбора эпохи."
            ),
            ""
        ]

        results = []
        all_error_details = []

        for model_name, model_file in models_to_test:

            accuracy, confusion, error_details = (
                evaluate_model_on_user_samples(
                    model_file,
                    samples
                )
            )

            for error in error_details:

                gallery_error = error.copy()
                gallery_error["model_name"] = model_name

                all_error_details.append(
                    gallery_error
                )

            results.append(
                (
                    model_name,
                    accuracy
                )
            )

            report.extend(
                build_user_test_report_section(
                    model_name,
                    model_file,
                    accuracy,
                    confusion,
                    error_details
                )
            )

            report.append("")

        if len(results) == 2:

            difference = (
                results[1][1]
                - results[0][1]
            )

            report.extend(
                [
                    "=" * 72,
                    "СРАВНЕНИЕ МОДЕЛЕЙ",
                    f"Базовая CNN: {results[0][1]:.2f}%",
                    (
                        "CNN после исправлений: "
                        f"{results[1][1]:.2f}%"
                    ),
                    (
                        "Изменение: "
                        f"{difference:+.2f} процентного пункта"
                    )
                ]
            )

        report_text = "\n".join(report)

        with open(
            TEST_REPORT_FILE,
            "w",
            encoding="utf-8"
        ) as report_file:

            report_file.write(report_text)

        gallery_error_count = create_error_gallery(
            all_error_details
        )

        summary = " | ".join(
            f"{name}: {accuracy:.2f}%"
            for name, accuracy in results
        )

        test_status.config(
            text=(
                f"✓ Проверка завершена. {summary}. "
                f"Отчёт: {TEST_REPORT_FILE}. "
                f"Ошибок на изображении: {gallery_error_count}."
            )
        )

        print()
        print(report_text)
        print()
        print(
            "Отчёт сохранён:",
            os.path.abspath(TEST_REPORT_FILE)
        )

        print(
            "Изображение ошибок сохранено:",
            os.path.abspath(TEST_ERRORS_IMAGE_FILE)
        )

    except Exception as error:

        test_status.config(
            text=(
                "Ошибка проверки тестового набора: "
                f"{error}"
            )
        )

        print(
            "Ошибка проверки тестового набора:",
            error
        )

    finally:

        set_training_controls(True)


# ============================================================
# 18. ОЧИСТКА ПОЛЯ
# ============================================================

def clear():

    global image
    global draw
    global last_x
    global last_y
    global last_mnist_image
    global last_prediction

    canvas.delete(
        "all"
    )

    image = Image.new(
        "L",
        (
            CANVAS_SIZE,
            CANVAS_SIZE
        ),
        color=0
    )

    draw = ImageDraw.Draw(
        image
    )

    last_x = None
    last_y = None
    last_mnist_image = None
    last_prediction = None

    correct_digit_var.set(
        ""
    )

    result_label.config(
        text="Нарисуйте цифру от 0 до 9"
    )

    confidence_label.config(
        text=""
    )

    probability_label.config(
        text=""
    )

    correction_status.config(
        text=""
    )

    test_status.config(
        text=""
    )


# ============================================================
# 19. ИНТЕРФЕЙС
# ============================================================

title_label = tk.Label(
    root,
    text="Распознавание рукописных цифр",
    font=(
        "Arial",
        24,
        "bold"
    )
)

title_label.pack(
    pady=(
        15,
        5
    )
)


main_frame = tk.Frame(
    root
)

main_frame.pack(
    fill=tk.BOTH,
    expand=True,
    padx=25,
    pady=10
)


left_frame = tk.Frame(
    main_frame
)

left_frame.pack(
    side=tk.LEFT,
    fill=tk.BOTH,
    expand=True,
    padx=15
)


# Правая часть содержит много элементов. Помещаем её в прокручиваемую
# область, чтобы нижний тестовый раздел был доступен на любом экране.
right_container = tk.Frame(
    main_frame
)

right_container.pack(
    side=tk.RIGHT,
    fill=tk.BOTH,
    expand=True,
    padx=15
)


right_scrollbar = tk.Scrollbar(
    right_container,
    orient=tk.VERTICAL
)

right_scrollbar.pack(
    side=tk.RIGHT,
    fill=tk.Y
)


right_scroll_canvas = tk.Canvas(
    right_container,
    highlightthickness=0,
    borderwidth=0,
    bg=root.cget("bg"),
    yscrollcommand=right_scrollbar.set
)

right_scroll_canvas.pack(
    side=tk.LEFT,
    fill=tk.BOTH,
    expand=True
)

right_scrollbar.config(
    command=right_scroll_canvas.yview
)


right_frame = tk.Frame(
    right_scroll_canvas,
    bg=root.cget("bg")
)

right_window = right_scroll_canvas.create_window(
    (0, 0),
    window=right_frame,
    anchor="nw"
)


def update_right_scroll_region(event=None):

    right_scroll_canvas.configure(
        scrollregion=right_scroll_canvas.bbox("all")
    )


def fit_right_panel_width(event):

    right_scroll_canvas.itemconfigure(
        right_window,
        width=event.width
    )


def scroll_right_panel(event):

    if event.delta == 0:
        return

    right_scroll_canvas.yview_scroll(
        int(-event.delta / 120),
        "units"
    )


def enable_right_mousewheel(event):

    root.bind_all(
        "<MouseWheel>",
        scroll_right_panel
    )


def disable_right_mousewheel(event):

    root.unbind_all(
        "<MouseWheel>"
    )


right_frame.bind(
    "<Configure>",
    update_right_scroll_region
)

right_scroll_canvas.bind(
    "<Configure>",
    fit_right_panel_width
)

right_scroll_canvas.bind(
    "<Enter>",
    enable_right_mousewheel
)

right_scroll_canvas.bind(
    "<Leave>",
    disable_right_mousewheel
)


draw_title = tk.Label(
    left_frame,
    text="Нарисуйте одну цифру мышкой",
    font=(
        "Arial",
        15,
        "bold"
    )
)

draw_title.pack(
    pady=(
        0,
        8
    )
)


canvas = tk.Canvas(
    left_frame,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    cursor="cross",
    highlightthickness=1,
    highlightbackground="gray"
)

canvas.pack(
    pady=5
)


button_frame = tk.Frame(
    left_frame
)

button_frame.pack(
    pady=12
)


predict_button = tk.Button(
    button_frame,
    text="Распознать",
    command=predict,
    font=(
        "Arial",
        13,
        "bold"
    ),
    width=15,
    height=2
)

predict_button.pack(
    side=tk.LEFT,
    padx=6
)


clear_button = tk.Button(
    button_frame,
    text="Очистить",
    command=clear,
    font=(
        "Arial",
        13
    ),
    width=15,
    height=2
)

clear_button.pack(
    side=tk.LEFT,
    padx=6
)


result_title = tk.Label(
    right_frame,
    text="Результат",
    font=(
        "Arial",
        15,
        "bold"
    )
)

result_title.pack(
    pady=(
        10,
        15
    )
)


result_label = tk.Label(
    right_frame,
    text="Нарисуйте цифру от 0 до 9",
    font=(
        "Arial",
        22,
        "bold"
    )
)

result_label.pack(
    pady=(
        5,
        8
    )
)


confidence_label = tk.Label(
    right_frame,
    text="",
    font=(
        "Arial",
        16
    )
)

confidence_label.pack(
    pady=5
)


probability_label = tk.Label(
    right_frame,
    text="",
    font=(
        "Consolas",
        11
    ),
    justify=tk.LEFT
)

probability_label.pack(
    pady=(
        10,
        20
    )
)


separator = tk.Frame(
    right_frame,
    height=2,
    bg="lightgray"
)

separator.pack(
    fill=tk.X,
    pady=10
)


correction_title = tk.Label(
    right_frame,
    text="Модель зафиксирована",
    font=(
        "Arial",
        16,
        "bold"
    )
)

correction_title.pack(
    pady=(
        8,
        6
    )
)


correction_instruction = tk.Label(
    right_frame,
    text=(
        "В режиме final test распознавание, коррекция, дообучение "
        "и сброс модели отключены."
    ),
    font=(
        "Arial",
        11
    ),
    wraplength=520,
    justify=tk.CENTER
)

correction_instruction.pack(
    pady=5
)


correction_input_frame = tk.Frame(
    right_frame
)

correction_input_frame.pack(
    pady=10
)


correct_digit_label = tk.Label(
    correction_input_frame,
    text="Правильная цифра:",
    font=(
        "Arial",
        13
    )
)

correct_digit_label.pack(
    side=tk.LEFT,
    padx=6
)


correct_digit_var = (
    tk.StringVar()
)


correct_digit_entry = tk.Spinbox(
    correction_input_frame,
    from_=0,
    to=9,
    textvariable=correct_digit_var,
    width=4,
    justify="center",
    font=(
        "Arial",
        16
    )
)

correct_digit_entry.pack(
    side=tk.LEFT,
    padx=6
)


correction_button = tk.Button(
    right_frame,
    text="Исправить и дообучить",
    command=correct_and_train,
    font=(
        "Arial",
        12,
        "bold"
    ),
    width=24,
    height=2
)

correction_button.pack(
    pady=6
)


train_all_button = tk.Button(
    right_frame,
    text="Переобучить на всех исправлениях",
    command=train_on_all_corrections,
    font=(
        "Arial",
        11
    ),
    width=30,
    height=2
)

train_all_button.pack(
    pady=6
)


reset_model_button = tk.Button(
    right_frame,
    text="Сбросить пользовательское обучение",
    command=reset_user_model,
    font=(
        "Arial",
        10
    ),
    width=32
)

reset_model_button.pack(
    pady=6
)


correction_status = tk.Label(
    right_frame,
    text="",
    font=(
        "Arial",
        11
    ),
    wraplength=540,
    justify=tk.CENTER
)

correction_status.pack(
    pady=(
        8,
        6
    )
)


test_separator = tk.Frame(
    right_frame,
    height=2,
    bg="lightgray"
)

test_separator.pack(
    fill=tk.X,
    pady=6
)


test_title = tk.Label(
    right_frame,
    text="Независимый final test",
    font=(
        "Arial",
        14,
        "bold"
    )
)

test_title.pack(
    pady=(
        4,
        3
    )
)


test_instruction = tk.Label(
    right_frame,
    text=(
        "Сначала выберите цифру и нарисуйте её, затем сразу сохраните. "
        "Нужно ровно 20 новых вариантов каждой цифры."
    ),
    font=(
        "Arial",
        10
    ),
    wraplength=540,
    justify=tk.CENTER
)

test_instruction.pack(
    pady=3
)


test_input_frame = tk.Frame(
    right_frame
)

test_input_frame.pack(
    pady=5
)


test_digit_label = tk.Label(
    test_input_frame,
    text="Нарисованная цифра:",
    font=(
        "Arial",
        11
    )
)

test_digit_label.pack(
    side=tk.LEFT,
    padx=4
)


test_digit_var = tk.StringVar(
    value="0"
)


test_digit_entry = tk.Spinbox(
    test_input_frame,
    from_=0,
    to=9,
    textvariable=test_digit_var,
    width=3,
    justify="center",
    font=(
        "Arial",
        13
    )
)

test_digit_entry.pack(
    side=tk.LEFT,
    padx=4
)


save_test_button = tk.Button(
    test_input_frame,
    text="Сохранить в final test",
    command=save_test_sample,
    font=(
        "Arial",
        10,
        "bold"
    ),
    width=24
)

save_test_button.pack(
    side=tk.LEFT,
    padx=6
)


evaluate_test_button = tk.Button(
    right_frame,
    text="Запустить финальную оценку",
    command=evaluate_user_test_dataset,
    font=(
        "Arial",
        10,
        "bold"
    ),
    width=34
)

evaluate_test_button.pack(
    pady=4
)


open_errors_button = tk.Button(
    right_frame,
    text="Открыть изображение ошибок",
    command=open_errors_image,
    font=(
        "Arial",
        10
    ),
    width=30
)

open_errors_button.pack(
    pady=3
)


test_counts_label = tk.Label(
    right_frame,
    text="",
    font=(
        "Consolas",
        9
    ),
    wraplength=540,
    justify=tk.CENTER
)

test_counts_label.pack(
    pady=3
)


test_status = tk.Label(
    right_frame,
    text="",
    font=(
        "Arial",
        9
    ),
    wraplength=540,
    justify=tk.CENTER
)

test_status.pack(
    pady=(
        2,
        5
    )
)


model_info_label = tk.Label(
    root,
    text="",
    font=(
        "Arial",
        10
    )
)

model_info_label.pack(
    pady=(
        0,
        12
    )
)


# ============================================================
# 20. СОБЫТИЯ
# ============================================================

canvas.bind(
    "<Button-1>",
    start_draw
)

canvas.bind(
    "<B1-Motion>",
    draw_digit
)

canvas.bind(
    "<ButtonRelease-1>",
    stop_draw
)


# Удобные клавиши:
# Enter = сохранить рисунок в final test
# Ctrl+L = очистить
def handle_enter(event):
    save_test_sample()


def handle_clear(event):
    clear()


root.bind(
    "<Return>",
    handle_enter
)

root.bind(
    "<Control-l>",
    handle_clear
)

root.bind(
    "<Control-L>",
    handle_clear
)


# ============================================================
# 21. ЗАПУСК
# ============================================================

update_model_info()
update_test_info()
set_training_controls(True)

root.mainloop()
