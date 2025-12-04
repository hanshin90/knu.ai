import streamlit as st
# import cv2
from PIL import Image


st.title('Hello, kitty')
st.write('this is for deplying at streamlit')
st.title('what is lenna?')
st.write("Lenna (or Lena) is a standard test image used in the field of digital image processing, starting in 1973.[1] It is a picture of the Swedish model Lena Forsén, shot by photographer Dwight Hooker and cropped from the centerfold of the November 1972 issue of Playboy magazine. The Lenna image has attracted controversy because of its subject matter.[2] Starting in the mid-2010s, many journals have deemed it inappropriate and discouraged its use, while others have banned it from publication outright.[3][4][5][6] Forsén herself has called for it to be retired, saying It's time I retired from tech.")

img = Image.open("Lenna_(test_image).png")

st.image(img)

