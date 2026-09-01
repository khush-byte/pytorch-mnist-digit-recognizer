import os
import random
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

import torch
from torch import nn
from PIL import Image, ImageDraw
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

os.makedirs(
    CORRECTIONS_DIR,
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
    "Распознавание цифр — PyTorch с обучением на исправлениях"
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


def set_training_controls(enabled):

    state = tk.NORMAL if enabled else tk.DISABLED

    predict_button.config(state=state)
    clear_button.config(state=state)
    correction_button.config(state=state)
    train_all_button.config(state=state)
    reset_model_button.config(state=state)
    correct_digit_entry.config(state=state)


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
# 17. ОЧИСТКА ПОЛЯ
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


# ============================================================
# 18. ИНТЕРФЕЙС
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


right_frame = tk.Frame(
    main_frame
)

right_frame.pack(
    side=tk.RIGHT,
    fill=tk.BOTH,
    expand=True,
    padx=15
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
    text="Коррекция модели",
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
        "Если AI ошибся, укажите правильную цифру "
        "и нажмите «Исправить и дообучить»."
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
        10
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
# 19. СОБЫТИЯ
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
# Enter = распознать
# Ctrl+L = очистить
def handle_enter(event):
    predict()


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
# 20. ЗАПУСК
# ============================================================

update_model_info()

root.mainloop()
