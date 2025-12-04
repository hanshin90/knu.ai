import streamlit as st
# import cv2
from PIL import Image


st.title('Hello, kitty')
st.write('this is for deplying at streamlit')
st.title('what is lenna?')
st.write("Lenna (or Lena) is a standard test image used in the field of digital image processing, starting in 1973.[1] It is a picture of the Swedish model Lena Forsén, shot by photographer Dwight Hooker and cropped from the centerfold of the November 1972 issue of Playboy magazine. The Lenna image has attracted controversy because of its subject matter.[2] Starting in the mid-2010s, many journals have deemed it inappropriate and discouraged its use, while others have banned it from publication outright.[3][4][5][6] Forsén herself has called for it to be retired, saying It's time I retired from tech.")

img = Image.open("Lenna_(test_image).png")

st.image(img)

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===============================
# 1. Hyperparameters
# ===============================
batch_size = 64
lr = 0.001
epochs = 5

# ===============================
# 2. MNIST Dataset & Loader
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # mean 0.5, std 0.5
])

train_data = datasets.MNIST(
    root=".",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root=".",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# ===============================
# 3. DNN Model
# ===============================
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = DNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ===============================
# 4. Training
# ===============================
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images = images.view(-1, 28*28)  # Flatten

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# ===============================
# 5. Save model
# ===============================
torch.save(model.state_dict(), "mnist_dnn.pth")
print("Model saved: mnist_dnn.pth")

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# -----------------------------
# 1. 모델 정의
# -----------------------------
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# 2. 모델 불러오기
# -----------------------------
model = DNN()
model.load_state_dict(torch.load("mnist_dnn.pth", map_location="cpu"))
model.eval()
print("Model loaded!")



# -------------------------
# 3. MNIST 테스트 세트 불러오기
# -------------------------
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)



# -------------------------
# 4. 이미지 1개 가져오기
# -------------------------
img, label = test_dataset[0]
print("True Label:", label)


# 이미지 표시
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.show()



# DNN input 형태로 변환 (1, 784)
img_flat = img.view(1, 28*28)

# -------------------------
# 5. 추론
# -------------------------
with torch.no_grad():
    output = model(img_flat)
    pred = torch.argmax(output, dim=1).item()

print("Predicted:", pred)
