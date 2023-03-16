import cv2
import time
import numpy as np
from LookUp import LookupTable
import math

def filter_outliers(data, center, threshold):
    for i in range(len(data)):
        if math.dist((data[i][0], data[i][2]), center) > threshold * np.std(data):
            data[i] = [-1, -1, -1, -1]

    return data[data[:, 0] > -1]

def create_table(x, y, z, voxel_size):
    table = LookupTable(x, y, z, voxel_size)
    return table


def get_voxels(table, masks):
    voxels, colors = table.get_voxels(masks)
    return voxels


def cluster(voxels, filter):
    coords = np.empty((len(voxels), 3), dtype=np.float32)
    coords2 = np.empty((len(voxels), 2), dtype=np.float32)
    for i, voxel in enumerate(voxels):
        x, y, z = voxel[0], voxel[1], voxel[2]
        coords[i] = np.array([x, y, z], dtype=np.float32)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(coords[:, [0, 2]], 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labeled_voxels = np.hstack((coords, label)).astype(int)

    clusters = [labeled_voxels[labeled_voxels[:, 3] == 0, :], labeled_voxels[labeled_voxels[:, 3] == 1, :],
                labeled_voxels[labeled_voxels[:, 3] == 2, :], labeled_voxels[labeled_voxels[:, 3] == 3, :]]

    if filter == 1:
        for i in range(4):
            clusters[i] = filter_outliers(clusters[i], center[i], 0.40)
        # merged = np.concatenate((clusters[0], clusters[1], clusters[2], clusters[3]))
        # clusters = cluster(merged, 0)

    return clusters


def create_colormodel(table, clusters, view, phase):
    # List which will at the end contain the four color models
    color_models = []

    # Loop over all voxel clusters
    for i in range(len(clusters)):

        # Lists to store image points per view
        image_points1 = []
        image_points2 = []
        image_points3 = []
        image_points4 = []

        # Per voxel in the clusters, store the image points per view
        for j in range(len(clusters[i])):
            image_points1.append(table.voxel2coord[clusters[i][j][0], clusters[i][j][1], clusters[i][j][2], 0])
            image_points2.append(table.voxel2coord[clusters[i][j][0], clusters[i][j][1], clusters[i][j][2], 1])
            image_points3.append(table.voxel2coord[clusters[i][j][0], clusters[i][j][1], clusters[i][j][2], 2])
            image_points4.append(table.voxel2coord[clusters[i][j][0], clusters[i][j][1], clusters[i][j][2], 3])

        # Final list of four lists of image points
        image_points = [image_points1, image_points2, image_points3, image_points4]

        # List to store histograms
        histograms = []

        if phase == "Online":
            # Loop over the four views
            for n in range(4):

                # Obtain the colored frame of that view and convert it to HSV
                frame = cv2.cvtColor(cv2.imread(f'data/cam{n + 1}/offline_phase/{view}_colored.png'), cv2.COLOR_BGR2HSV)

                # Initialize a mask, first all zeroes
                mask = np.zeros_like(frame)

                # Loop over the corresponding list of image points, and assign 1 to that point in the mask
                for m in range(len(image_points[n])):
                    mask[image_points[n][m][1], image_points[n][m][0]] = \
                        frame[image_points[n][m][1], image_points[n][m][0]]

                # Add the resulting histogram to the list of histograms
                histograms.append(cv2.calcHist(mask, [0, 1], None, [180, 256], [0, 180, 0, 256]))

            # The color model is the average HS-histogram of the four views
            final_histogram = (histograms[0] + histograms[1] + histograms[2] + histograms[3]) / 4
            color_models.append(final_histogram)

        if phase == "Offline":
            # Obtain the colored frame of that view and convert it to HSV
            frame = cv2.imread(f'data/cam{view}/offline_phase/{view}_colored.png')
            # frame = cv2.cvtColor(cv2.imread(f'data/cam{view}/offline_phase/{view}_colored.png'), cv2.COLOR_BGR2HSV)

            # Initialize a mask, first all zeroes
            mask = np.zeros_like(frame)

            # Loop over the corresponding list of image points, and assign 1 to that point in the mask
            for m in range(len(image_points[view - 1])):
                mask[image_points[view - 1][m][1], image_points[view - 1][m][0]] = \
                    frame[image_points[view - 1][m][1], image_points[view - 1][m][0]]

            cv2.imshow('mask', mask)
            cv2.waitKey(0)

            histogram = cv2.calcHist([mask], [0, 1], None, [180, 256], [0, 180, 0, 256])
            color_models.append(histogram)

    return color_models

# Offline phase
def offline_phase():
    # Create lookup table
    # table = create_table(136, 188, 80, 25)
    table = create_table(68, 94, 40, 50)
    print("\n")

    unmatched_color_models = []
    matched_color_models = []
    final_color_models = []

    # For every camera view
    for i in range(1, 5):
        print(f"Offline phase: view {i}/4")

        # Prepare the needed frames
        frame1, mask1 = cv2.imread(f'data/cam1/offline_phase/{i}_colored.png'), cv2.imread(
            f'data/cam1/offline_phase/{i}_fgmask.png')
        frame2, mask2 = cv2.imread(f'data/cam2/offline_phase/{i}_colored.png'), cv2.imread(
            f'data/cam2/offline_phase/{i}_fgmask.png')
        frame3, mask3 = cv2.imread(f'data/cam3/offline_phase/{i}_colored.png'), cv2.imread(
            f'data/cam3/offline_phase/{i}_fgmask.png')
        frame4, mask4 = cv2.imread(f'data/cam4/offline_phase/{i}_colored.png'), cv2.imread(
            f'data/cam4/offline_phase/{i}_fgmask.png')

        # Make the 3D-reconstruction
        voxels = get_voxels(table, [cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY), cv2.cvtColor(mask4, cv2.COLOR_BGR2GRAY)])

        # Cluster the voxels
        clusters = cluster(voxels, 1)

        # Determine the color model of the projection to the specified camera image, add them to the list
        color_models_per_view = create_colormodel(table, clusters, i, "Offline")

        unmatched_color_models.append(color_models_per_view)
        print("\n")

    # Functionality to match color models based on histogram difference
    for i in range(4):
        best_match = [unmatched_color_models[0][i]]
        for j in range(1, 4):
            best_match.append(unmatched_color_models[j][0])
            for n in range(4):
                difference = cv2.compareHist(unmatched_color_models[0][i], unmatched_color_models[j][n], 1)
                if difference < cv2.compareHist(unmatched_color_models[0][i], best_match[j], 1):
                    best_match[j] = unmatched_color_models[j][n]
        matched_color_models.append(best_match)

    # Average the matched histograms into one color model
    for i in range(4):
        average_histogram = (matched_color_models[i][0] + matched_color_models[i][1] + matched_color_models[i][2] +
                             matched_color_models[i][3]) / 4
        final_color_models.append(average_histogram)

    return final_color_models
offline_phase()
