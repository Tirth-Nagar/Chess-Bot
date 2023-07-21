import os
import cv2
import uuid
import numpy as np

def fen_to_matrix(fen_str):
    # Initialize an empty 8x8 chessboard matrix
    board = [['E' for _ in range(8)] for _ in range(8)]

    # Row and column indices to fill the board matrix
    row, col = 0, 0

    # Iterate through the FEN positions and fill the board matrix
    for char in fen_str:
        if char == '-':
            row += 1
            col = 0
        elif char.isdigit():
            # Empty squares indicated by a number, so move the specified number of columns forward
            col += int(char)
        else:
            # Non-empty square, place the piece on the board
            board[row][col] = char
            col += 1

    return np.array(board)

def split_chessboard(chessboard_img, fen_matrix):
    rows = 8
    cols = 8

    position_height = chessboard_img.shape[0] // rows
    position_width = chessboard_img.shape[1] // cols
    
    output_folder = 'test/'
    
    for row in range(rows):
        for col in range(cols):
            
            output_path = output_folder+str(fen_matrix[row,col])
            os.makedirs(output_path, exist_ok=True)        
            
            y_start = row * position_height
            y_end = (row + 1) * position_height
            
            x_start = col * position_width
            x_end = (col + 1) * position_width
            
            position_img = chessboard_img[y_start:y_end, x_start:x_end]
            
            output_filename = os.path.join(output_path, f'{str(uuid.uuid4())}.png')
            
            cv2.imwrite(output_filename, position_img)
            
            
chessboard_img = cv2.imread("dataset/test/1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg")
fen_str = "1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8"

fen_matrix = fen_to_matrix(fen_str)
split_chessboard(chessboard_img, fen_matrix)