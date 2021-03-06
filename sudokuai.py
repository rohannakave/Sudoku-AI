import streamlit as st
import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform


def read_image(image_path):
  image = cv2.imread(image_path)
  image = imutils.resize(image, width=600)
  # cv2.imshow("Image", image)
  # cv2.waitKey(0)
  st.image(image)
  return image

def find_puzzle(image, debug=False):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (7,7),3)
  thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
  thresh = cv2.bitwise_not(thresh)

  if debug:
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    st.image(thresh)
  
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
  
  puzzleCnt = None

  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
      puzzleCnt = approx
      break
      if puzzleCnt is None:
        raise Exception(("Could not find sudoku puzzle outline. "
        "Try debugging your thresholding and contour steps."))
  if debug:
    output = image.copy()
    cv2.drawContours(output, [puzzleCnt], -1, (0,255,0), 2)
    # cv2.imshow("output", output)
    # cv2.waitKey(0)
    st.image(output)
  
  puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
  warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

  if debug:
    # cv2.imshow("puzzle", puzzle)
    # cv2.waitKey(0)
    st.image(puzzle)
  
  return (puzzle, warped)

def extract_digit(thresh, debug=False):
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  if cnts == []:
    digit = "."
    return digit  
  c = max(cnts, key = cv2.contourArea)
  mask = np.zeros(thresh.shape, dtype = 'uint8')
  cv2.drawContours(mask, [c], -1, 255, -1)
  (h, w) = thresh.shape
  percentFilled = cv2.countNonZero(mask) / float(w * h)
  if percentFilled < 0.03:
    digit = "."
    return digit
  digit = cv2.bitwise_and(thresh, thresh, mask=mask)
  if debug:
    # cv2.imshow("Digits", digit)
    # cv2.waitKey(0)
    st.image(digit)
  return digit