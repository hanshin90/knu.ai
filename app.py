import streamlit as st
# import cv2
from PIL import Image


st.title('Hello, kitty')
st.write('this is for deplying at streamlit')
st.title('what is lenna?')
st.write("Lenna (or Lena) is a standard test image used in the field of digital image processing, starting in 1973.[1] It is a picture of the Swedish model Lena Forsén, shot by photographer Dwight Hooker and cropped from the centerfold of the November 1972 issue of Playboy magazine. The Lenna image has attracted controversy because of its subject matter.[2] Starting in the mid-2010s, many journals have deemed it inappropriate and discouraged its use, while others have banned it from publication outright.[3][4][5][6] Forsén herself has called for it to be retired, saying It's time I retired from tech.")

img = Image.open("Lenna_(test_image).png")

st.image(img)

import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# 1. DNN Model
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
# Streamlit UI
# -----------------------------
st.title("MNIST Digit Classification with DNN (PyTorch + Streamlit)")
st.write("학습 / 모델 불러오기 / 테스트셋 이미지 예측을 수행할 수 있는 DEMO입니다.")


# -----------------------------
# 2. Train Model
# -----------------------------
if st.button("Train Model"):
    st.write("Training Started...")

    batch_size = 64
    lr = 0.001
    epochs = 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = DNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.view(-1, 28*28)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        st.write(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "mnist_dnn.pth")
    st.success("Model saved: mnist_dnn.pth")


# -----------------------------
# 3. Load Model
# -----------------------------
if st.button("Load Model"):
    model = DNN()
    model.load_state_dict(torch.load("mnist_dnn.pth", map_location="cpu"))
    model.eval()
    st.success("Model Loaded!")


# -----------------------------
# 4. Predict From MNIST Testset
# -----------------------------
if st.button("Predict Test Image"):
    st.write("Loading MNIST Test Dataset...")

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )

    img, label = test_dataset[0]

    st.write(f"**True Label: {label}**")
    st.image(img.squeeze().numpy(), width=200, caption="MNIST Test Image", clamp=True)

    img_flat = img.view(1, 28*28)

    model = DNN()
    model.load_state_dict(torch.load("mnist_dnn.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(img_flat)
        pred = torch.argmax(output, dim=1).item()

    st.write(f"### ✅ Predicted: **{pred}**")
