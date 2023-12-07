# chess-position-detector

This program takes in an image of a chess board and outputs the FEN position, which is a standardized notation to describe a chess position. It will also show a visualization of the position. The FEN notation can then be used to analyze the position with a chess engine or to set up a position on a virtual chess board.

The jupyter notebook `intermediate.ipynb` also contains intermediate visualizations of each of the steps, run on a sample image. See also our [Google Drive](https://drive.google.com/drive/folders/1E6TVYuCSJiE1tBr730CTBkAQF9aNq9Ug) for storing the object detection results (everything except for the data and JSON results are already stored in this repository).

## Usage

To use the program, run the following command:

```
python chess_fen.py --image <path_to_image>
```

## Example

```
python chess_fen.py --image sample_images/3828.png
```

This will output the FEN position and show a visualization of the position.

## Installation

To install the required packages, run the following command:

```
pip install -r requirements.txt
```
