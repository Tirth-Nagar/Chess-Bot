import os
import cv2
import numpy as np
from keras.models import load_model
import pyautogui as pag
import pygetwindow as gw

def find_chessboard(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Ensure the polygon has four corners
    if len(approx_polygon) == 4:
        # Order the corners clockwise starting from the top-left
        chessboard_corners = np.array([approx_polygon[0][0], approx_polygon[1][0], approx_polygon[2][0], approx_polygon[3][0]], dtype=np.float32)
        ordered_corners = np.zeros((4, 2), dtype=np.float32)

        sums = chessboard_corners.sum(axis=1)
        diffs = np.diff(chessboard_corners, axis=1)

        ordered_corners[0] = chessboard_corners[np.argmin(sums)]
        ordered_corners[1] = chessboard_corners[np.argmin(diffs)]
        ordered_corners[2] = chessboard_corners[np.argmax(sums)]
        ordered_corners[3] = chessboard_corners[np.argmax(diffs)]

        return ordered_corners

    return None

def split_chessboard(chessboard):
    
    rows = 8
    cols = 8

    position_height = chessboard.shape[0] // rows
    position_width = chessboard.shape[1] // cols
    
    output_folder = 'current_positions'
    os.makedirs(output_folder, exist_ok=True)

    for row in range(rows):
        for col in range(cols):
            y_start = row * position_height
            y_end = (row + 1) * position_height
            
            x_start = col * position_width
            x_end = (col + 1) * position_width
            
            position_img = chessboard[y_start:y_end, x_start:x_end]
            
            output_filename = os.path.join(output_folder, f'{row}{col}.png')
            
            cv2.imwrite(output_filename, position_img)
            
def get_screenshot():         
    # Get window by title
    window_title = 'Play Chess Online for FREE with Friends - Chess.com'
    window = gw.getWindowsWithTitle(window_title)[0]

    # Put the window in focus
    window.activate()

    # Get the position and size of the window
    x, y, width, height = window.left, window.top, window.width, window.height

    # Take a screenshot of the window
    screenshot = pag.screenshot(region=(x, y, width, height))

    # Save the screenshot
    screenshot.save('screenshot.png')

    # Load the screenshot image
    screenshot = cv2.imread('screenshot.png')

    # Find the chessboard in the screenshot
    chessboard_corners = find_chessboard(screenshot)

    if chessboard_corners is not None:
        # Perform perspective transformation to obtain a top-down view of the chessboard
        width, height = 400, 400  # Set the desired output size of the chessboard
        destination_corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        transformation_matrix = cv2.getPerspectiveTransform(chessboard_corners, destination_corners)
        transformed_chessboard = cv2.warpPerspective(screenshot, transformation_matrix, (width, height))

        split_chessboard(transformed_chessboard)

    else:
        print('Chessboard not found in the screenshot.')


def predict_piece(image_path):
    # Load the trained model
    model = load_model('chess_piece_classifier.h5')  # Replace 'chess_piece_classifier.h5' with the actual path to your saved model

    subfolders = ['b', 'E', 'k', 'n', 'p', 'q', 'r', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']

    # Preprocess the sample image
    sample_img = cv2.imread(image_path)
    sample_img = cv2.resize(sample_img, (50,50))
    sample_img = np.expand_dims(sample_img, axis=0)
    sample_img = sample_img / 255.0  # Normalize pixel values to [0, 1]

    # Make the prediction using the model
    predictions = model.predict(sample_img)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = subfolders[predicted_class_index]

    return predicted_class_label.strip('w')

def predict_board():
    board = [['' for i in range(8)] for j in range(8)]
    for i in range(8):
        for j in range(8):
            image_path = f'current_positions/{i}{j}.png'
            board[i][j] = predict_piece(image_path)
    return np.array(board)
        
def board_to_fen(board):
    fen = ""
    for row in board:
        empty_count = 0
        for square in row:
            if square == 'E':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += square
        if empty_count > 0:
            fen += str(empty_count)
        fen += '/'

    # Remove the trailing slash at the end of the last row
    fen = fen[:-1]

    return fen


# get_screenshot()
print(board_to_fen(predict_board()))
