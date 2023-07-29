# Chess-Bot
 Chess-Bot is a project that focuses on developing a highly capable chess-playing bot designed specifically for playing on Chess.com. This project aims to leverage computer vision and the Stockfish Chess Engine to create an intelligent and strategic chess player that can compete against human opponents on the popular online chess platform.

## Overview

The Chess Piece Classifier is a deep learning project that aims to classify images of chess pieces into different categories based on their type and color. The project uses a convolutional neural network (CNN) to achieve accurate and robust classification.

## Dataset

The training dataset consists of images of chessboards, each of size 400x400 pixels with each image file labelled as the associated FEN string. The dataset can be found [here](https://www.kaggle.com/datasets/koryakinp/chess-positions) for free.

The associated data_processing.py file allows me to take and split each image into 64 images and sort them into folders according to the FEN string provided.

- Black Pieces are stored in folders named:
  - {Piece Type} (Folder Name) 
  - Black Bishop (b)
  - Black King (k)
  - Black Knight (n)
  - Black Pawn (p)
  - Black Queen (q)
  - Black Rook (r)

- White Pieces are stored in folders named:
  - {Piece Type} (Folder Name)
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
   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;git clone https://github.com/your_username/chess-piece-classifier.git

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cd chess-piece-classifier

2. Install the required dependencies (ensure you have Python 3.x and pip installed):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pip install -r requirements.txt

3. Download the dataset and pre-process the data using the instructions located in the data_processing.py file
4. Run the main.py file for the best move possible!

