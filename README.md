# Chess-Bot
 ChessBot is a project that focuses on developing a highly capable chess-playing bot designed specifically for playing on Chess.com. This project aims to leverage computer vision and the Stockfish Chess Engine to create an intelligent and strategic chess player that can compete against human opponents on the popular online chess platform.


![Chess Piece Classifier](chess_piece_classifier.png)

## Overview

The Chess Piece Classifier is a deep learning project that aims to classify images of chess pieces into different categories based on their type and color. The project uses a convolutional neural network (CNN) to achieve accurate and robust classification.

## Dataset

The training dataset consists of images of chess pieces, each of size 50x50 pixels. The dataset is organized into subfolders for different classes:

- Black Pieces:
  - Black Bishop (b)
  - Black King (k)
  - Black Knight (n)
  - Black Pawn (p)
  - Black Queen (q)
  - Black Rook (r)

- White Pieces (prefixed with 'w'):
  - White Bishop (wB)
  - White King (wK)
  - White Knight (wN)
  - White Pawn (wP)
  - White Queen (wQ)
  - White Rook (wR)

- Empty Square (E)

## Model Architecture

The CNN model architecture used in this project is as follows:

1. Convolutional layer with 32 filters, kernel size (3, 3), and ReLU activation.
2. Max pooling layer with pool size (2, 2).
3. Convolutional layer with 64 filters, kernel size (3, 3), and ReLU activation.
4. Max pooling layer with pool size (2, 2).
5. Flatten layer to convert the 2D output to a 1D vector.
6. Dense layer with 64 units and ReLU activation.
7. Output layer with softmax activation for multi-class classification.

The model is compiled using the Adam optimizer and categorical cross-entropy loss function.

## Getting Started

1. Clone this repository to your local machine:

git clone https://github.com/your_username/chess-piece-classifier.git
cd chess-piece-classifier


2. Install the required dependencies (ensure you have Python 3.x and pip installed):

pip install -r requirements.txt

4. Run the main.py file


