import cv2
import numpy as np
from matplotlib import pyplot as plt
from LookUp import LookupTable
from scipy.optimize import linear_sum_assignment
import math


def find_clusters(voxel_data, filter):
    # Prepare the voxel data for the kmeans function
    voxels = np.column_stack((voxel_data[0], voxel_data[1], voxel_data[2]))

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(voxels[:, 0:2].astype(np.float32), 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    data = np.append(voxels, label, axis=1)

    data = filter_outliers(data, center, 4)[:, 0:3]

    _, label, center = cv2.kmeans(data[:, 0:2].astype(np.float32), 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    data = np.append(data, label, axis=1)
    # Filter outliers (NOT FINISHED)
    #if filter == 1:
        #for i in range(4):
            #data[data[:, 3] == i] = filter_outliers(data[data[:, 3] == i], center[i], 0.5)
            # data = find_clusters([data[:, 0], data[:, 1], data[:, 2]], 0)
        #data = data[data[:, 0] > -1]

    return data, center


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
            mask = np.zeros(image.shape[0:2], np.uint8)
            #mask[x_coords[:, n], y_coords[:, n]] = image[x_coords[:, n], y_coords[:, n]]
            mask[x_coords[:, n], y_coords[:, n]] = 255
            kernel = np.ones((3,3), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations = 2)
            #mask[mask[:, :, 1] == 255] =

            # Show image points used for histogram
            #cv2.imshow("img", mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Calculate histogram
            #cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([image_hsv], [0, 1], mask, [16, 16], [0, 180, 60, 256]
                                , accumulate=False)

            # Normalize histogram
            total_sum_bins = np.sum(hist)
            hist = hist / total_sum_bins

            # Save histogram
            histograms[n, m] = hist

    return histograms


def show_histograms(histograms):
    fig, axs = plt.subplots(2, 2)
    counter = -1
    for i in range(2):
        for j in range(2):
            counter += 1
            axs[i, j].plot(histograms[counter], color='b')

    plt.show()


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


def hungarian_algorithm(distances):
    # An array to store the best matches
    best_matches = np.zeros((4, 4, 4), dtype=np.uint8)

    # Calculating the best matches for all four views
    for image in np.arange(0, 4):
        row, col = linear_sum_assignment(distances[image])
        best_matches[image, row, col] = 1

    # Find the best matches of the four views combined
    joint = np.sum(best_matches, axis=0)*-1
    col, row = linear_sum_assignment(joint)

    return col

def adjust_labels(voxel_data, labels):
    data_copy = np.copy(voxel_data)

    for label in np.arange(4):
        voxel_data[data_copy[:, 3] == label][:, 3] = labels[label]

    return voxel_data


def get_colors(voxel_data):
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
    voxel_colors = np.empty((voxel_data.shape[0], 3), dtype=np.uint8)

    for n in np.arange(0, 4):
        voxel_colors[voxel_data[:, 3] == n] = colors[n, :]

    return voxel_colors.tolist()


def filter_outliers(data, center, threshold):
    sets = []
    for i in range(4):
        subset = data[data[:, 3] == i]
        distances = np.sqrt(np.square(subset[:, 0] - center[i, 0]) + np.square(subset[:, 1] - center[i, 1]))
        std = np.std(distances)

        filter = distances < threshold * std
        subset = subset[filter, :]
        sets.append(subset)

    result = np.concatenate(tuple(sets), axis=0)

    return result

def tracking():
    pass


