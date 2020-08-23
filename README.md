# Sudoku-AI
Steps Overview:-

Sudoku Solver Web app using Streamlit.
Computer Vision is used to detect the boundaries of sudoku puzzle in image and also to detect the digits as well as position of digits in the puzzle.
CNN is used to predict the detected digits in the puzzle.
After predicting the digits they were placed in 9*9 size numpy array.
The vaccant places in the puzzle is replaced by 0.
Then using rules of sudoku puzzle game and using backtracking algorithm we get the solution of sudoku puzzle.
