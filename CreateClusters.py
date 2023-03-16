import cv2
import numpy as np
from matplotlib import pyplot as plt
from LookUp import LookupTable

def find_clusters(coords):

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(coords, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    data = np.append(coords, label, axis = 1)
    return data, center

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
data, colors, voxels_on = table.get_voxels([mask1, mask2, mask3, mask4])

voxels_on_array = np.column_stack((voxels_on[0], voxels_on[1], voxels_on[2]))
voxels_on_array = voxels_on_array[np.where(voxels_on_array[:,2] >= 18)]
data, centers = find_clusters(voxels_on_array.astype(np.float32))

def getHistograms(color_images, voxel_data, table):
    # An array to store the histograms
    histograms = np.empty((4, 4, 180, 256))

    # Computing the histograms
    for n, image in enumerate(color_images):
        for m in np.arange(0, 4):
            # Get the voxels of cluster m
            voxel_cluster = voxel_data[np.where(voxel_data[:, 3] == m)].astype(int)

            # Find image points of the voxels in all four images
            x_coords = table.voxel2coord[data_[:,0], data_[:,1], data_[:,2], :, 1]
            y_coords = table.voxel2coord[data_[:,0], data_[:,1], data_[:,2], :, 0]

            # Create mask
            mask = np.zeros(image.shape[0:2], np.uint8)
            mask[x_coords[:, n], y_coords[:, n]] = 255

            # Show image points used for histogram
            #cv2.imshow("img", mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Calculate histogram
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([image_hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])

            # Save histogram
            histograms[n, m] = hist

    return histograms

images = [img1, img2, img3, img4]
for i, img in enumerate(images):
    histograms = []
    for j in range(4):
        data_ = data[np.where(data[:, 3] == j)].astype(int)

        colors_y = table.voxel2coord[data_[:,0], data_[:,1], data_[:,2], :, 0]
        colors_x = table.voxel2coord[data_[:,0], data_[:,1], data_[:,2], :, 1]

        # Create mask
        mask = np.zeros(img.shape[0:2], np.uint8)
        mask[colors_x[:, i], colors_y[:, i]] = 255

        # Show image points used for histogram
        cv2.imshow("img", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Calculate histogram
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        print(hist.shape)
        # Save histogram
        histograms.append(hist)

    # plot the above computed histogram
    fig, axs = plt.subplots(2, 2)
    t = -1
    for p in range(2):
        for q in range(2):
            t += 1
            axs[p, q].plot(histograms[t], color='b')

    plt.show()

comparison_a = cv2.compareHist(histograms[0], histograms[1], cv2.HISTCMP_CHISQR)
comparison_b = cv2.compareHist(histograms[0], histograms[2], cv2.HISTCMP_CHISQR)
comparison_c = cv2.compareHist(histograms[0], histograms[3], cv2.HISTCMP_CHISQR)
comparison_d = cv2.compareHist(histograms[1], histograms[2], cv2.HISTCMP_CHISQR)
comparison_e = cv2.compareHist(histograms[1], histograms[3], cv2.HISTCMP_CHISQR)
comparison_f = cv2.compareHist(histograms[2], histograms[3], cv2.HISTCMP_CHISQR)

print(comparison_a)
print(comparison_b)
print(comparison_c)
print(comparison_d)
print(comparison_e)
print(comparison_f)