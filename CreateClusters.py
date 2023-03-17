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

    data = filter_outliers(data, center, 0.5)[:, 0:3]

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
            mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            mask[x_coords[:, n], y_coords[:, n]] = 255
            # kernel = np.ones((3, 3), dtype=np.uint8)
            # mask = cv2.dilate(mask, kernel, iterations=1)
            # mask = cv2.erode(mask, kernel, iterations=1)

            # mask2 = np.zeros_like(image)
            # for x in np.arange(0, 486):
            #     for y in np.arange(0, 644):
            #         if mask[x, y]== 255:
            #             mask2[x, y] = image[x, y]
            #
            # # Show image points used for histogram
            # cv2.imshow("img", mask2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

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
        col, row = linear_sum_assignment(distances[image])
        best_matches[image, row, col] = 1

    # Find the best matches of the four views combined
    joint = np.sum(best_matches, axis=0)*-1
    row, col = linear_sum_assignment(joint)

    return col

# TODO: Hij maakt nu alle waarden in voxel_data hetzelfde als het label, bijv [7, 13, 21, 1] wordt [1, 1, 1, 1]
def adjust_labels(voxel_data, labels):
    temp_values = [10, 11, 12, 13]
    for i in np.arange(0, 4):
        voxel_data[voxel_data[:, 3] == i] = temp_values[i]
    for i in np.arange(0, 4):
        voxel_data[voxel_data[:, 3] == i + 10] = labels[i]
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
        print(subset.shape)
        #distances = np.sqrt(np.square(subset[:, 0] - center[i, 0]) + np.square(subset[:, 1] - center[i, 1]))

        dx = np.sqrt(np.square(subset[:, 0] - center[i, 0]))
        dy = np.sqrt(np.square(subset[:, 1] - center[i, 1]))

        std_x = np.std(subset[:, 0])
        std_y = np.std(subset[:, 1])

        filter_x = dx < threshold * std_x
        filter_y = dy < threshold * std_y
        filter = np.logical_or(filter_x, filter_y)
        subset = subset[filter, :]
        sets.append(subset)

    result = np.concatenate(tuple(sets), axis=0)


    #for i in range(len(data)):
        #if math.dist((data[i][0], data[i][1]), center) > threshold * np.std(data):
            #data[i] = [-1, -1, -1, -1]

    return result

"""
####################
img1 = cv2.imread("data/cam1/color.png")
img2 = cv2.imread("data/cam2/color.png")
img3 = cv2.imread("data/cam3/color.png")
img4 = cv2.imread("data/cam4/color.png")

mask1 = cv2.imread("data/cam1/voxel.png", cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread("data/cam2/voxel.png", cv2.IMREAD_GRAYSCALE)
mask3 = cv2.imread("data/cam3/voxel.png", cv2.IMREAD_GRAYSCALE)
mask4 = cv2.imread("data/cam4/voxel.png", cv2.IMREAD_GRAYSCALE)

voxel_size = 50
table = LookupTable(68, 94, 40, voxel_size)
data1, voxels_on = table.get_voxels([mask1, mask2, mask3, mask4])

data, centers = find_clusters(voxels_on)
histograms = get_histograms([img1, img2, img3, img4], data, table)

altered_histograms = np.array([histograms[:, 3, :, :], histograms[:, 1, :, :], histograms[:, 0, :, :], histograms[:, 2, :, :]])

distances = calculate_distances(histograms, histograms)
labels = hungarian_algorithm(distances)
voxel_data = adjust_labels(data, labels)
c = get_colors(voxel_data)

for i in range(4):
    show_histograms(histograms[i])
"""


