import cv2
import numpy as np
from matplotlib import pyplot as plt
from LookUp import LookupTable
from scipy.optimize import linear_sum_assignment

def find_clusters(voxel_data):
    # Prepare the voxel data for the kmeans function
    voxels = np.column_stack((voxel_data[0], voxel_data[1], voxel_data[2]))

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(voxels.astype(np.float32), 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    data = np.append(voxels, label, axis=1)

    return data, center

def get_histograms(color_images, voxel_data, table):
    # An array to store the histograms
    histograms = np.empty((4, 4, 180, 256), dtype=np.float32)

    voxel_data = voxel_data[voxel_data[:, 2] >= 18]
    voxel_data = voxel_data[voxel_data[:, 2] <= 29]

    # Computing the histograms
    for n, image in enumerate(color_images):
        for m in np.arange(0, 4):
            # Get the voxels of cluster m
            voxel_cluster = voxel_data[voxel_data[:, 3] == m].astype(int)

            # Find image points of the voxels in all four images
            x_coords = table.voxel2coord[voxel_cluster[:,0], voxel_cluster[:,1], voxel_cluster[:,2], :, 1]
            y_coords = table.voxel2coord[voxel_cluster[:,0], voxel_cluster[:,1], voxel_cluster[:,2], :, 0]

            # Create mask
            mask = np.zeros(image.shape, np.uint8)
            mask[x_coords[:, n], y_coords[:, n]] = image[x_coords[:, n], y_coords[:, n]]

            # Show image points used for histogram
            #cv2.imshow("img", mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Calculate histogram
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([image_hsv], [0, 1], cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), [180, 256], [0, 180, 0, 256])

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
                distances[image, row, column] = cv2.compareHist(ground_truth[image, row, :, :], histograms[image, column, :, :],
                                                         cv2.HISTCMP_CHISQR)

    return distances

def hungarian_algorithm(distances):
    # An array to store the best matches
    best_matches = np.zeros((4,4,4), dtype=np.uint8)

    # Calculating the best matches for all four views
    for image in np.arange(0, 4):
        row, col = linear_sum_assignment(distances[image])
        best_matches[image, row, col] = 1

    # Find the best matches of the four views combined
    joint = np.sum(best_matches, axis = 0)*-1
    row, col = linear_sum_assignment(joint)

    return col

def adjust_labels(voxel_data, labels):
    for i in np.arange(0, 4):
        voxel_data[voxel_data[:, 3] == i] = labels[i]
    return voxel_data

def get_colors(voxel_data):
    colors = np.array([[255, 0, 0],[0, 255, 0],[0, 0, 255],[255, 255, 255]])
    voxel_colors = np.empty((voxel_data.shape[0], 3), dtype=np.uint8)

    for n in np.arange(0, 4):
        voxel_colors[voxel_data[:, 3] == n] = colors[n, :]

    return voxel_colors.tolist()

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


