import sys
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Zdefiniujmy transformacje dla obrazów
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),  # Odwrócenie kolorów
])

# Zdefiniujmy etykiety klas
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Wczytaj obrazek
try:
    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

# Definiujmy klasy modeli
model_paths_adam = {
    "model_v1.pth": "Model v1 Adam",
    "model_v2.pth": "Model v2 Adam",
    "model_v3.pth": "Model v3 Adam",
}

model_paths_sgd = {
    "model_v1_sgd.pth": "Model v1 SGD",
    "model_v2_sgd.pth": "Model v2 SGD",
    "model_v3_sgd.pth": "Model v3 SGD",
}

# Klasy modeli
class NeuralNetwork_v1(nn.Module):
    def __init__(self):
        super(NeuralNetwork_v1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_v2(nn.Module):
    def __init__(self):
        super(NeuralNetwork_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_v3(nn.Module):
    def __init__(self):
        super(NeuralNetwork_v3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Klasy modeli z optymalizatorem SGD
class NeuralNetwork_v1_sgd(NeuralNetwork_v1):
    pass


class NeuralNetwork_v2_sgd(NeuralNetwork_v2):
    pass


class NeuralNetwork_v3_sgd(NeuralNetwork_v3):
    pass


# Mapa klas modeli do ich definicji
model_classes = {
    "model_v1.pth": NeuralNetwork_v1,
    "model_v2.pth": NeuralNetwork_v2,
    "model_v3.pth": NeuralNetwork_v3,
    "model_v1_sgd.pth": NeuralNetwork_v1_sgd,
    "model_v2_sgd.pth": NeuralNetwork_v2_sgd,
    "model_v3_sgd.pth": NeuralNetwork_v3_sgd,
}

# Funkcja do klasyfikacji i wyświetlania wyników
def classify_and_display(model_paths, model_name_prefix):
    for model_path, model_name in model_paths.items():
        try:
            # Inicjalizujmy model
            model_class = model_classes[model_path]
            model = model_class()

            # Wczytajmy wagi modelu
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Przeprowadźmy klasyfikację
            with torch.no_grad():
                pred = model(image_tensor)
                probabilities = torch.nn.functional.softmax(pred, dim=1).squeeze()
                predicted = torch.argmax(probabilities).item()

            # Wyświetlmy wynik
            print(f"{model_name}: {classes[predicted]} (Prawdopodobieństwo: {probabilities[predicted]:.2f})")

            # Wyświetlmy prawdopodobieństwa predykcji dla każdej klasy
            plt.figure(figsize=(10, 5))
            plt.bar(classes, probabilities.numpy())
            plt.xticks(rotation=45)
            plt.title(f"Prawdopodobieństwa predykcji dla każdej klasy - {model_name}")
            plt.show()
        except Exception as e:
            print(f"Error with model {model_name}: {e}")

# Klasyfikacja i wyświetlanie wyników dla modeli Adam
print("\nWyniki dla modeli z optymalizatorem Adam:")
classify_and_display(model_paths_adam, "Model Adam")

# Klasyfikacja i wyświetlanie wyników dla modeli SGD
print("\nWyniki dla modeli z optymalizatorem SGD:")
classify_and_display(model_paths_sgd, "Model SGD")

# Wyświetlmy obraz wejściowy z nałożoną etykietą predykcji dla modelu v3 Adam
model_path = "model_v3.pth"
model_name = model_paths_adam[model_path]
try:
    model_class = model_classes[model_path]
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        pred = model(image_tensor)
        probabilities = torch.nn.functional.softmax(pred, dim=1).squeeze()
        predicted = torch.argmax(probabilities).item()

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f"Predykcja: {classes[predicted]} (Prawdopodobieństwo: {probabilities[predicted]:.2f}) - {model_name}")
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Error with final model {model_name}: {e}")

# Wyświetlmy obraz wejściowy z nałożoną etykietą predykcji dla modelu v3 SGD
model_path = "model_v3_sgd.pth"
model_name = model_paths_sgd[model_path]
try:
    model_class = model_classes[model_path]
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        pred = model(image_tensor)
        probabilities = torch.nn.functional.softmax(pred, dim=1).squeeze()
        predicted = torch.argmax(probabilities).item()

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f"Predykcja: {classes[predicted]} (Prawdopodobieństwo: {probabilities[predicted]:.2f}) - {model_name}")
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Error with final model {model_name}: {e}")