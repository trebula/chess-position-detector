import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import pdist, squareform


def _absolute_angle_difference(x, y):
    diff = np.mod(np.abs(x - y), 2 * np.pi)
    # angles are cyclical so return smaller of the two possible differences
    return np.min(np.stack([diff, np.pi - diff], axis=-1), axis=-1)


def get_intersections(horizontal_lines, vertical_lines):
    # get intersection points
    all_intersections = []
    for hline in horizontal_lines:
        for vline in vertical_lines:
            rho1, theta1 = hline
            rho2, theta2 = vline
            x, y = _get_intersection(rho1, theta1, rho2, theta2)
            all_intersections.append((x, y))
    all_intersections = np.array(all_intersections)
    all_intersections = np.concatenate([all_intersections, np.ones((*all_intersections.shape[:-1], 1))], axis=-1)
    return all_intersections


def _get_intersection(rho1, theta1, rho2, theta2):
    cos_t1 = np.cos(theta1)
    cos_t2 = np.cos(theta2)
    sin_t1 = np.sin(theta1)
    sin_t2 = np.sin(theta2)
    x = (sin_t1 * rho2 - sin_t2 * rho1) / (cos_t2 * sin_t1 - cos_t1 * sin_t2)
    y = (cos_t1 * rho2 - cos_t2 * rho1) / (sin_t2 * cos_t1 - sin_t1 * cos_t2)
    return x, y


def _fix_negative_rho(lines: np.ndarray) -> np.ndarray:
    lines = lines.copy()
    neg_rho_mask = lines[..., 0] < 0
    lines[neg_rho_mask, 0] = -lines[neg_rho_mask, 0]
    lines[neg_rho_mask, 1] = lines[neg_rho_mask, 1] - np.pi
    return lines


def get_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 90, 400)
    return cv2.HoughLines(edged, 1, np.pi / 360, 100)


def compute_vertical_and_horizontal_lines(lines):
    # cluster lines based on their angles
    lines = lines.squeeze(axis=-2)
    lines = _fix_negative_rho(lines)
    thetas = lines[..., 1].reshape(-1, 1)
    distance_matrix = pdist(thetas, metric=_absolute_angle_difference)
    distance_matrix = squareform(distance_matrix)

    clustering = AgglomerativeClustering(metric="precomputed", linkage="complete", n_clusters=2)
    clusters = clustering.fit_predict(distance_matrix)

    # determine which cluster is the horizontal lines and which is the vertical lines
    angle_with_y_axis = _absolute_angle_difference(thetas, 0.0)
    if angle_with_y_axis[clusters == 0].mean() > angle_with_y_axis[clusters == 1].mean():
        hcluster, vcluster = 0, 1
    else:
        hcluster, vcluster = 1, 0

    horizontal_lines = lines[clusters == hcluster]
    vertical_lines = lines[clusters == vcluster]
    return horizontal_lines, vertical_lines


def cluster_similar_lines(horizontal_lines, vertical_lines):
    # find mean vertical line
    mean_vertical_rho, mean_vertical_theta = vertical_lines.mean(axis=0)
    rho, theta = np.moveaxis(horizontal_lines, -1, 0)
    intersections = _get_intersection(rho, theta, mean_vertical_rho, mean_vertical_theta)
    intersections = np.stack(intersections, axis=-1)

    clustering = DBSCAN(eps=12, min_samples=1)
    clusters = clustering.fit(intersections)

    # filter horizontal lines
    filtered_lines = []
    for c in range(clusters.labels_.max() + 1):
        lines_in_cluster = horizontal_lines[clusters.labels_ == c]
        rho = lines_in_cluster[..., 0]
        median = np.argsort(rho)[len(rho) // 2]
        filtered_lines.append(lines_in_cluster[median])
    horizontal_lines = np.stack(filtered_lines, axis=0)

    # find mean horizontal line
    mean_horizontal_rho, mean_horizontal_theta = horizontal_lines.mean(axis=0)
    rho, theta = np.moveaxis(vertical_lines, -1, 0)
    intersections = _get_intersection(rho, theta, mean_horizontal_rho, mean_horizontal_theta)
    intersections = np.stack(intersections, axis=-1)

    clusters = clustering.fit(intersections)

    # filter vertical lines
    filtered_lines = []
    for c in range(clusters.labels_.max() + 1):
        lines_in_cluster = vertical_lines[clusters.labels_ == c]
        rho = lines_in_cluster[..., 0]
        median = np.argsort(rho)[len(rho) // 2]
        filtered_lines.append(lines_in_cluster[median])
    vertical_lines = np.stack(filtered_lines, axis=0)

    return horizontal_lines, vertical_lines


def computeOptimalHomography(all_intersections, horizontal_lines, vertical_lines, square_size):
    iterations = 0
    best_inliers = []
    best_candidate_mapping = []
    while iterations < 200 or len(best_inliers) < 30:
        # randomly select 2 horizontal and 2 vertical lines
        h_rand_indices = np.random.choice(len(horizontal_lines), 2, replace=False)
        h_rand_indices = np.sort(h_rand_indices)
        v_rand_indices = np.random.choice(len(vertical_lines), 2, replace=False)
        v_rand_indices = np.sort(v_rand_indices)
        rand_horizontal = horizontal_lines[h_rand_indices]
        rand_vertical = vertical_lines[v_rand_indices]

        rand_horizontal = rand_horizontal[np.argsort(rand_horizontal[..., 0])]
        rand_vertical = rand_vertical[np.argsort(rand_vertical[..., 0])]

        # get the intersection points
        top_horizontal = rand_horizontal[0]
        bottom_horizontal = rand_horizontal[1]
        left_vertical = rand_vertical[0]
        right_vertical = rand_vertical[1]
        intersections = [
            _get_intersection(top_horizontal[0], top_horizontal[1], left_vertical[0], left_vertical[1]),  # top left
            _get_intersection(top_horizontal[0], top_horizontal[1], right_vertical[0], right_vertical[1]),  # top right
            _get_intersection(
                bottom_horizontal[0], bottom_horizontal[1], right_vertical[0], right_vertical[1]
            ),  # bottom right
            _get_intersection(
                bottom_horizontal[0], bottom_horizontal[1], left_vertical[0], left_vertical[1]
            ),  # bottom left
        ]
        intersections = np.array(intersections)

        for s_x in range(1, 9):
            for s_y in range(1, 9):
                dst = np.array([[0, 0], [s_x, 0], [s_x, s_y], [0, s_y]], dtype=np.float32)
                H, _ = cv2.findHomography(intersections, dst)

                inliers = []
                warped_points = set()
                candidate_mapping = []
                for p in all_intersections:
                    # warp the point
                    p_prime = p @ H.T
                    p_prime = p_prime[..., :2] / p_prime[..., 2, np.newaxis]
                    q = np.round(p_prime).astype(int)
                    if tuple(q) in warped_points:
                        continue
                    # L1 norm
                    error = np.sum(np.abs(p_prime - q))
                    if error < 0.05:
                        inliers.append(p)
                        warped_points.add(tuple(q))
                        candidate_mapping.append((p, q))
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_candidate_mapping = candidate_mapping
        if iterations >= 500:
            break
        iterations += 1

    intersections = np.array([p for p, _ in best_candidate_mapping])
    warped_points = np.array([q for _, q in best_candidate_mapping])
    # translate warped points
    warped_points = warped_points - warped_points.min(axis=0) + 5
    dst = warped_points * square_size
    # recompute the homography using the inliers
    H, _ = cv2.findHomography(intersections, dst)
    return best_inliers, warped_points, best_candidate_mapping, np.array(H, dtype=np.float32)


def warpImage(image, square_size, warped_points, H):
    max_size = (warped_points.max(axis=0) + 5) * square_size
    warped = cv2.warpPerspective(image, H, (max_size[0], max_size[1]))
    return warped


def inferMissingLines(warped, warped_points, best_inliers, best_candidate_mapping, square_size):
    # infer missing lines
    threshold = 2
    w_x_min = warped_points[:, 0].min()
    w_x_max = warped_points[:, 0].max()
    w_y_min = warped_points[:, 1].min()
    w_y_max = warped_points[:, 1].max()
    w_x_diff = w_x_max - w_x_min
    w_y_diff = w_y_max - w_y_min

    # infer vertical lines
    while w_x_diff > 8:
        # remove all points corresponding to the left/right columns
        remove_indices = np.where((warped_points[:, 0] == w_x_min) | (warped_points[:, 0] == w_x_max))[0]
        warped_points = np.delete(warped_points, remove_indices, axis=0)
        best_inliers = np.delete(best_inliers, remove_indices, axis=0)
        best_candidate_mapping = np.delete(best_candidate_mapping, remove_indices, axis=0)
        w_x_min = warped_points[:, 0].min()
        w_x_max = warped_points[:, 0].max()
        w_x_diff = w_x_max - w_x_min
    while w_x_diff < 8:
        # horizontal sobel filter
        sobelx = np.abs(cv2.Sobel(warped.copy(), cv2.CV_64F, 1, 0, ksize=3))
        vertical_edges = cv2.Canny(sobelx.astype(np.uint8), 250, 500)
        # sum pixels on the vertical lines to the left and right of min and max x
        left_mid = (w_x_min - 1) * square_size
        left_sum = np.sum(vertical_edges[:, left_mid - threshold : left_mid + threshold + 1])
        right_mid = (w_x_max + 1) * square_size
        right_sum = np.sum(vertical_edges[:, right_mid - threshold : right_mid + threshold + 1])
        if left_sum > right_sum:
            # add left column
            w_x_min -= 1
        else:
            # add right column
            w_x_max += 1
        w_x_diff = w_x_max - w_x_min

    # infer horizontal lines
    while w_y_diff > 8:
        # remove all points corresponding to the top/bottom rows
        remove_indices = np.where((warped_points[:, 1] == w_y_min) | (warped_points[:, 1] == w_y_max))[0]
        warped_points = np.delete(warped_points, remove_indices, axis=0)
        best_inliers = np.delete(best_inliers, remove_indices, axis=0)
        best_candidate_mapping = np.delete(best_candidate_mapping, remove_indices, axis=0)
        w_y_min = warped_points[:, 1].min()
        w_y_max = warped_points[:, 1].max()
        w_y_diff = w_y_max - w_y_min
    while w_y_diff < 8:
        # vertical sobel filter
        sobely = np.abs(cv2.Sobel(warped.copy(), cv2.CV_64F, 0, 1, ksize=3))
        horizontal_edges = cv2.Canny(sobely.astype(np.uint8), 250, 500)
        # sum pixels on the horizontal lines to the top and bottom of min and max y
        top_mid = (w_y_min - 1) * square_size
        top_sum = np.sum(horizontal_edges[top_mid - threshold : top_mid + threshold, :])
        bottom_mid = (w_y_max + 1) * square_size
        bottom_sum = np.sum(horizontal_edges[bottom_mid - threshold : bottom_mid + threshold, :])
        if top_sum > bottom_sum:
            # add top row
            w_y_min -= 1
        else:
            # add bottom row
            w_y_max += 1
        w_y_diff = w_y_max - w_y_min
    return w_x_min, w_x_max, w_y_min, w_y_max


def findAllIntersection(w_x_min, w_x_max, w_y_min, w_y_max, square_size):
    intersections = np.zeros((9, 9, 2))
    for row in range(w_y_min, w_y_max + 1):
        for col in range(w_x_min, w_x_max + 1):
            intersections[row - w_y_min, col - w_x_min] = [col * square_size, row * square_size]
    return intersections
