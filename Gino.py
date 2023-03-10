import cv2
import time
import numpy as np
from LookUp import LookupTable

# Masks used for initialization (frame 0 of each view)
mask1 = cv2.imread('data/cam1/voxel.png', cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread('data/cam2/voxel.png', cv2.IMREAD_GRAYSCALE)
mask3 = cv2.imread('data/cam3/voxel.png', cv2.IMREAD_GRAYSCALE)
mask4 = cv2.imread('data/cam4/voxel.png', cv2.IMREAD_GRAYSCALE)


def create_table(x, y, z):
    start = time.time()
    table = LookupTable(x, y, z)
    end = time.time()
    print(f"Execution time lookup table creation: {(end - start)} seconds")
    return table


def get_voxels(table, masks):
    start = time.time()
    voxels, colors = table.get_voxels(masks)
    end = time.time()
    print(f"Execution time get_voxels: {(end - start)} seconds")
    return voxels


def cluster(voxels):
    start = time.time()
    coords = np.empty((len(voxels), 3), dtype=np.float32)
    for i, voxel in enumerate(voxels):
        x, y, z = voxel[0], voxel[1], voxel[2]
        coords[i] = np.array([x, y, z], dtype=np.float32)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(coords, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labeled_voxels = np.hstack((coords, label)).astype(int)

    clusters = [labeled_voxels[labeled_voxels[:, 3] == 0, :], labeled_voxels[labeled_voxels[:, 3] == 1, :],
                labeled_voxels[labeled_voxels[:, 3] == 2, :], labeled_voxels[labeled_voxels[:, 3] == 3, :]]

    end = time.time()
    print(f"Execution time clustering: {(end - start)} seconds")
    return clusters


def create_colormodel(table, clusters):

    start = time.time()

    # List which will at the end contain the four color models
    color_models = []

    # Loop over all voxel clusters
    for i in range(len(clusters)):

        # Lists to store image points per view
        image_points1 = image_points2 = image_points3 = image_points4 = []

        # Per voxel in the clusters, store the image points per view
        for j in range(len(clusters[i])):
            image_points1.append(table.voxel2coord[clusters[i][j][0]][clusters[i][j][1]][clusters[i][j][2]][0])
            image_points2.append(table.voxel2coord[clusters[i][j][0]][clusters[i][j][1]][clusters[i][j][2]][1])
            image_points3.append(table.voxel2coord[clusters[i][j][0]][clusters[i][j][1]][clusters[i][j][2]][2])
            image_points4.append(table.voxel2coord[clusters[i][j][0]][clusters[i][j][1]][clusters[i][j][2]][3])

        # Final list of four lists of image points
        image_points = [image_points1, image_points2, image_points3, image_points4]

        # List to store histograms
        histograms = []

        # Loop over the four views
        for n in range(4):

            # Obtain the colored frame of that view and convert it to HSV
            frame = cv2.cvtColor(cv2.imread(f'data/cam{n + 1}/color.png'), cv2.COLOR_BGR2HSV)

            # Initialize a mask, first all zeroes
            mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # Loop over the corresponding list of image points, and assign 1 to that point in the mask
            for m in range(len(image_points[n])):
                mask[image_points[n][m][1], image_points[n][m][0]] = 1

            # Add the obtained histogram to the list of histograms
            histograms.append(cv2.calcHist([frame], [0, 1], mask, [180, 256], [0, 180, 0, 256]))

        # The color model is the average HS-histogram of the four views
        final_histogram = (histograms[0] + histograms[1] + histograms[2] + histograms[3]) / 4
        color_models.append(final_histogram)

    end = time.time()
    print(f"Execution time color models: {(end - start)} seconds")
    return color_models


# Offline phase: create table, get voxels based on first masks, cluster them, make color models of them
table = create_table(50, 50, 50)
voxels = get_voxels(table, [mask1, mask2, mask3, mask4])
clusters = cluster(voxels)
color_models = create_colormodel(table, clusters)

# TEST: Distance between histograms
d11 = cv2.compareHist(color_models[0], color_models[0], 1)
d22 = cv2.compareHist(color_models[1], color_models[1], 1)
d33 = cv2.compareHist(color_models[2], color_models[2], 1)
d44 = cv2.compareHist(color_models[3], color_models[3], 1)

d12 = cv2.compareHist(color_models[0], color_models[1], 1)
d13 = cv2.compareHist(color_models[0], color_models[2], 1)
d14 = cv2.compareHist(color_models[0], color_models[3], 1)
d23 = cv2.compareHist(color_models[1], color_models[2], 1)
d24 = cv2.compareHist(color_models[1], color_models[3], 1)
d34 = cv2.compareHist(color_models[2], color_models[3], 1)
