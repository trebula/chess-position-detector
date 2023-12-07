import cv2
import argparse
import matplotlib.pyplot as plt

from grid_detection import (
    get_lines,
    compute_vertical_and_horizontal_lines,
    cluster_similar_lines,
    get_intersections,
    computeOptimalHomography,
    warpImage,
    inferMissingLines,
    findAllIntersection,
)
from piece_detection import load_model, predict
from generate_fen import create_board, generate_FEN, point_transformation, label_pieces, visualize_fen


def main(args):
    SQUARE_SIZE = 50
    image = cv2.imread(args["image"])

    lines = get_lines(image)
    horizontal_lines, vertical_lines = compute_vertical_and_horizontal_lines(lines)
    horizontal_lines, vertical_lines = cluster_similar_lines(horizontal_lines, vertical_lines)

    all_intersections = get_intersections(horizontal_lines, vertical_lines)

    best_inliers, warped_points, best_candidate_mapping, H = computeOptimalHomography(
        all_intersections, horizontal_lines, vertical_lines, SQUARE_SIZE
    )

    warped = warpImage(image, SQUARE_SIZE, warped_points, H)

    w_x_min, w_x_max, w_y_min, w_y_max = inferMissingLines(
        warped,
        warped_points,
        best_inliers,
        best_candidate_mapping,
        SQUARE_SIZE,
    )

    intersections = findAllIntersection(w_x_min, w_x_max, w_y_min, w_y_max, SQUARE_SIZE)

    model = load_model()
    positions, labels = predict(model, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    warped_points = point_transformation(positions, H)
    pieces = label_pieces(warped_points, labels)
    board = create_board(intersections, pieces)
    fen = generate_FEN(board)

    print(f"FEN: {fen}")
    visualize_fen(fen)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    main(args)
