import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Устройство:", device)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


model = NeuralNetwork().to(device)

model.load_state_dict(
    torch.load(
        "mnist_model.pth",
        map_location=device
    )
)

model.eval()

print("Модель загружена.")


test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


index = 100

image, real_label = test_data[index]

input_image = image.unsqueeze(0).to(device)


with torch.no_grad():

    logits = model(input_image)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_label = probabilities.argmax(
        dim=1
    ).item()


print()
print("Настоящая цифра:", real_label)
print("Предсказание AI:", predicted_label)

print()
print("Вероятности:")

for digit in range(10):

    probability = probabilities[0][digit].item() * 100

    print(
        f"{digit}: {probability:.4f}%"
    )


plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(
    f"Real: {real_label} | AI: {predicted_label}"
)

plt.axis("off")

plt.show()