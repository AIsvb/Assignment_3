# Computer Vision: Assignment 3
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 18-03-2023

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import math

"""
This file contains functions for clustering voxel data, creating color models, and matching color models.
"""

# A function for clustering the voxel data
def find_clusters(voxel_data, filter):
    # Prepare the voxel data for the kmeans function
    voxels = np.column_stack((voxel_data[0], voxel_data[1], voxel_data[2]))

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(voxels[:, 0:2].astype(np.float32), 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    data = np.append(voxels, label, axis=1)

    # Filter outliers
    if filter == 1:
        for i in range(4):
            data[data[:, 3] == i] = filter_outliers(data[data[:, 3] == i], center[i], 0.5)
        data = data[data[:, 0] > -1]

    # Cluster again to correct the cluster centers
    _, label, center = cv2.kmeans(data[:, 0:2].astype(np.float32), 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    data = np.append(data, label, axis=1)

    return data, center


# A function for calculating the histograms for all clusters in all views
def get_histograms(color_images, voxel_data, table):
    # An array to store the histograms
    histograms = np.empty((4, 4, 16, 16), dtype=np.float32)

    # Remove trouser and head voxels
    voxel_data = voxel_data[voxel_data[:, 2] >= 18]
    voxel_data = voxel_data[voxel_data[:, 2] <= 29]

    # Computing the histograms
    for n, image in enumerate(color_images):
        for m in np.arange(0, 4):
            # Get the voxels of cluster with label m
            voxel_cluster = voxel_data[voxel_data[:, 3] == m].astype(int)

            # Find image points of the voxels in all four images
            x_coords = table.voxel2coord[voxel_cluster[:, 0], voxel_cluster[:, 1], voxel_cluster[:, 2], :, 1]
            y_coords = table.voxel2coord[voxel_cluster[:, 0], voxel_cluster[:, 1], voxel_cluster[:, 2], :, 0]

            # Create mask
            mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            mask[x_coords[:, n], y_coords[:, n]] = 255

            # Show image points used for histogram
            #cv2.imshow("img", mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Calculate histogram
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([image_hsv], [0, 1], mask, [16, 16], [0, 180, 120, 256]
                                , accumulate=False)

            # Normalize histogram
            total_sum_bins = np.sum(hist)
            if total_sum_bins > 0:
                hist = hist / total_sum_bins

            # Save histogram
            histograms[n, m] = hist

    return histograms


# A (helper) function for showing histograms
def show_histograms(histograms):
    fig, axs = plt.subplots(2, 2)
    counter = -1
    for i in range(2):
        for j in range(2):
            counter += 1
            axs[i, j].plot(histograms[counter], color='b')

    plt.show()


# A function for calculating the chi square distance between histograms
def calculate_distances(ground_truth, histograms):
    # An array to store all the distances
    distances = np.empty((4, 4, 4), dtype=np.float32)

    # Calculating the distances
    for image in np.arange(0, 4):
        for row in np.arange(0, 4):
            for column in np.arange(0, 4):
                distances[image, row, column] = cv2.compareHist(ground_truth[image, row, :, :],
                                                                histograms[image, column, :, :], cv2.HISTCMP_CHISQR)

    return distances


# A function that implements the Hungarian Algorithm to find the best matches of histograms with reference histograms
def hungarian_algorithm(distances):
    # An array to store the best matches
    best_matches = np.zeros((4, 4, 4), dtype=np.float32)

    # Calculating the best matches for all four views
    for image in np.arange(0, 4):
        col, row = linear_sum_assignment(distances[image])
        best_matches[image, row, col] = 1   # Saving the best matches

    # Find the best matches of the four views combined by means of majority voting
    joint = np.sum(best_matches, axis=0)*-1 # Multiply by -1, because linear_sum_assignment minimizes
    col, row = linear_sum_assignment(joint)

    return row


# A function that adjusts the labels of the clustered voxels, to make their labels match the labels of the
# reference clusters.
def adjust_labels(voxel_data, labels):
    data_copy = np.copy(voxel_data)

    for label in np.arange(4):
        data_copy[voxel_data[:, 3] == label, 3] = labels[label]

    return data_copy


# A function to compute the colors of the voxels given their labels
def get_colors(voxel_data):
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])     # The four colors
    voxel_colors = np.empty((voxel_data.shape[0], 3), dtype=np.uint8)               # Array to store the colors

    for n in np.arange(0, 4):
        voxel_colors[voxel_data[:, 3] == n] = colors[n, :]

    return voxel_colors.tolist()


# A function for filtering ghost voxels
def filter_outliers(data, center, threshold):
    for i in np.arange(0, len(data)):
        if math.dist((data[i][0], data[i][1]), center) > threshold * np.std(data):
            data[i] = [-1, -1, -1, -1]

    return data


