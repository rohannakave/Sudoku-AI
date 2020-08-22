import streamlit as st
import os
from sudokuai import extract_digit, find_puzzle, read_image
from sudoku import solve, print_board, find_empty
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
from skimage import exposure
from skimage.segmentation import clear_border

path = None
directory = os.getcwd()

# File Uploader widget 
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.sidebar.file_uploader("Upload a Image file")
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)  
    filename = 'savedImage.jpg'
    os.chdir(directory + '/' + 'puzzles')
    cv2.imwrite(filename, opencv_image) 
    path = os.getcwd() + '\\' + filename
    path = path.replace("\\", "/")

os.chdir(directory)

model_path = directory + '/' + 'digit_recognizer1.h5'
model_path = model_path.replace("\\", "/")

# Checkbox widget
ticked = False
agree = st.sidebar.checkbox("Click the checkbox to see the images of processing steps of Sudoku Solver.")
if agree:
  ticked = True


model = load_model(model_path)
if path is not None:
  st.write("Sudoku puzzle Image:-")
  image = read_image(image_path=path)

  (puzzleImage, warped) = find_puzzle(image, ticked)

  board = np.zeros((9, 9), dtype="int")
  stepX = warped.shape[1] // 9
  stepY = warped.shape[0] // 9
  cellLocs = []

  for y in range(0, 9):
    row=[]
    for x in range(0, 9):
      startX = x * stepX
      startY = y * stepY
      endX = (x + 1) * stepX
      endY = (y + 1) * stepY
      row.append((startX, startY, endX, endY))
      cell = warped[startY:endY, startX:endX]
      cell = exposure.rescale_intensity(cell, out_range = (0, 255))
      cell = cell.astype("uint8")
      thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      thresh = clear_border(thresh)
      digit = extract_digit(thresh, ticked)
      if digit != ".":
        roi = cv2.resize(digit, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        pred = model.predict(roi).argmax(axis=1)[0]
        #print(pred)
        board[y, x] = pred
    cellLocs.append(row)

  solve(board)

  r = st.sidebar.slider("R-value", 0, 255)
  g = st.sidebar.slider("G-value", 0, 255)
  b = st.sidebar.slider("B-value", 0, 255)

  for (cellRow, boardRow) in zip(cellLocs, board):
    for (box, digit) in zip(cellRow, boardRow):
      startX, startY, endX, endY = box
      textX = int((endX - startX) * 0.33)
      textY = int((endY - startY) * -0.2)
      textX += startX
      textY += endY
      cv2.putText(puzzleImage, str(digit), (textX, textY),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (r, g, b), 2)
  st.write("Answer of Sudoku puzzle Image:-")
  st.image(puzzleImage)
else:
  st.write("Upload the Sudoku Image")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)