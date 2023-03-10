import cv2
import numpy as np


def create_colormodel(table, clusters):
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

    return color_models

