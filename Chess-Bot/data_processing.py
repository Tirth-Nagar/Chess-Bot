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
            
            if char.isupper():
                # White piece
                board[row][col] = "w"+char
            else:
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

def get_x_files(folder_path, x):
    # Initialize an empty list to store the filenames
    file_list = []

    # Get the list of all files in the folder
    all_files = os.listdir(folder_path)

    # Iterate through each file and add it to the list, up to the specified number (x)
    count = 0
    for file in all_files:
        if count < x:
            file_list.append(file)
            count += 1
        else:
            break

    return file_list 


input_path = "dataset/test/"
file_list = get_x_files(input_path, 1000)
# print(file_list) 

for file in file_list:
    chessboard_img = cv2.imread(input_path+file)
    fen_str = file.split(".")[0]
    fen_matrix = fen_to_matrix(fen_str)
    split_chessboard(chessboard_img, fen_matrix)
