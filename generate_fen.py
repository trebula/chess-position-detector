import numpy as np
import matplotlib.pyplot as plt
from fentoboardimage import fenToImage, loadPiecesFolder


def point_transformation(position, H):
    # H is a 3*3 transformation matrix
    n = len(position)
    points = np.hstack((position, np.ones((n, 1))))
    transformed_points = np.dot(points, H.T)
    transformed_points[:, 0] /= transformed_points[:, 2]
    transformed_points[:, 1] /= transformed_points[:, 2]
    result_points = transformed_points[:, :2]
    return result_points


def label_pieces(position, labels):
    label_map = {1: "r", 2: "n", 3: "b", 4: "k", 5: "q", 6: "p", 7: "R", 8: "N", 9: "B", 10: "K", 11: "Q", 12: "P"}
    outputs = []
    for i in range(len(position)):
        output = []
        output.append(label_map[(labels[i])])
        output.append(position[i][0])
        output.append(position[i][1])
        outputs.append(output)
    return outputs


def create_board(intersections, pieces):
    """
    create_board creates 8x8 numpy array as chess board from intersection coordinates and piece coordinates


    :param intersections: 9x9 array of coordinate points that represent corners of boxes
    :param pieces: list of lists of length 3 of the form [[PIECE, x1, y1], [PIECE, x2, y2]...]
    :return: 8x8 numpy array with pieces in correct locations
    """
    pieces_dict = {(item[1], item[2]): item[0] for item in pieces}
    board = np.full((8, 8), "", dtype=object)
    for row_idx, row in enumerate(board):
        for col_idx, col in enumerate(row):
            top_left = intersections[row_idx][col_idx]
            top_right = intersections[row_idx][col_idx + 1]
            bot_left = intersections[row_idx + 1][col_idx]
            bot_right = intersections[row_idx + 1][col_idx + 1]
            for coords in pieces_dict:
                if (
                    coords[0] > top_left[0]
                    and coords[0] <= top_right[0]
                    and coords[1] > top_left[1]
                    and coords[1] <= bot_right[1]
                ):
                    board[row_idx, col_idx] = pieces_dict.get(coords)
                    break
    return board


def generate_FEN(board):
    """
    generate_FEN creates FEN code from 8x8 chess board list


    :param board: 8x8 chessboard with either a piece name or empty string
    :return: FEN code
    """
    count = 0
    output = ""
    for row_idx, row in enumerate(board):
        for col_idx, col in enumerate(row):
            if col != "":
                if count != 0:
                    output += str(count)
                    count = 0
                output += col
            else:
                count += 1
        if count != 0:
            output += str(count)
        if row_idx != len(board) - 1:
            output += "/"
        count = 0
    return output


def visualize_fen(fen):
    board = fenToImage(
        fen, squarelength=100, pieceSet=loadPiecesFolder("./pieces"), darkColor="#D18B47", lightColor="#FFCE9E"
    )
    plt.imshow(board)
    plt.show()
