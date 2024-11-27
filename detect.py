# detect.py
import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Zdefiniujmy transformacje dla obrazów
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Zdefiniujmy model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Wczytajmy wagi modelu z flagą weights_only=True
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

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

# Wczytajmy obrazek
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Przeprowadźmy klasyfikację
with torch.no_grad():
    pred = model(image_tensor)
    probabilities = torch.nn.functional.softmax(pred, dim=1).squeeze()
    predicted = torch.argmax(probabilities).item()

# Wyświetlmy wynik
print(classes[predicted])

# Wyświetlmy prawdopodobieństwa predykcji dla każdej klasy
plt.bar(classes, probabilities.numpy())
plt.xticks(rotation=45)
plt.title("Prawdopodobieństwa predykcji dla każdej klasy")
plt.show()

# Wyświetlmy obraz wejściowy z nałożoną etykietą predykcji
plt.imshow(image, cmap='gray')
plt.title(f"Predykcja: {classes[predicted]} (Prawdopodobieństwo: {probabilities[predicted]:.2f})")
plt.axis('off')
plt.show()