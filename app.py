import streamlit as st
# import cv2
from PIL import Image


st.title('Hello, kitty')
st.write('this is for deplying at streamlit')


img = Image.open("Lenna.png")

st.image(img)



@requirements.txt;

streamlit
opencv-python-headless
numpy
torch
