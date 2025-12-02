import streamlit as st
# import cv2
from PIL import Image


st.title('Hello, kitty')
st.write('this is for deplying at streamlit')

img = Image.open("Lenna_(test_image).png")

st.image(img)
