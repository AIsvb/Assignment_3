import glm
import numpy as np
import cv2
from LookUp import LookupTable as LT
import time
from CreateClusters import *

foreground1 = cv2.VideoCapture("data/cam1/foreground_cropped.avi")
foreground2 = cv2.VideoCapture("data/cam2/foreground_cropped.avi")
foreground3 = cv2.VideoCapture("data/cam3/foreground_cropped.avi")
foreground4 = cv2.VideoCapture("data/cam4/foreground_cropped.avi")

video1 = cv2.VideoCapture("data/cam1/video_cropped.avi")
video2 = cv2.VideoCapture("data/cam2/video_cropped.avi")
video3 = cv2.VideoCapture("data/cam3/video_cropped.avi")
video4 = cv2.VideoCapture("data/cam4/video_cropped.avi")

voxel_size = 50
table = LT(68, 94, 40, voxel_size)
histograms = np.empty((4, 4, 180, 256), dtype=np.float32)

# voxel_size = 25
# table = LT(136, 188, 80, voxel_size)

frame_no = 0
block_size = 1

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([57/255,138/255,123/255])
    return data, colors

def set_voxel_positions():
    global histograms, table
    _, mask_1 = foreground1.read()
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)

    _, mask_2 = foreground2.read()
    mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2GRAY)

    _, mask_3 = foreground3.read()
    mask_3 = cv2.cvtColor(mask_3, cv2.COLOR_BGR2GRAY)

    _, mask_4 = foreground4.read()
    mask_4 = cv2.cvtColor(mask_4, cv2.COLOR_BGR2GRAY)

    ret, img1 = video1.read()
    ret, img2 = video2.read()
    ret, img3 = video3.read()
    ret, img4 = video4.read()

    data, voxels_on = table.get_voxels([mask_1, mask_2, mask_3, mask_4])

    cluster_data, _ = find_clusters(voxels_on)
    histograms = get_histograms([img1, img2, img3, img4], cluster_data, table)

    colors = get_colors(cluster_data)

    return data, colors

def set_voxel_positions2():
    global mask_1a, mask_2a, mask_3a, mask_4a

    data, colors = table.get_voxels([mask_1a, mask_2a, mask_3a, mask_4a])

    clusters = cluster(data)

    data2 = []
    colors2 = []

    for i in range(4):
        for j in range(len(clusters[i])):
            data2.append([clusters[i][j][0], clusters[i][j][1], clusters[i][j][2]])
            if i == 0:
                colors2.append([1, 0, 0])
            if i == 1:
                colors2.append([0, 1, 0])
            if i == 2:
                colors2.append([0, 0, 1])
            if i == 3:
                colors2.append([1, 1, 1])


    return data2, colors2

# Function to set voxels based on a XOR-mask
def set_voxel_positions_live():
    global histograms, table
    _, mask_1 = foreground1.read()
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)

    _, mask_2 = foreground2.read()
    mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2GRAY)

    _, mask_3 = foreground3.read()
    mask_3 = cv2.cvtColor(mask_3, cv2.COLOR_BGR2GRAY)

    _, mask_4 = foreground4.read()
    mask_4 = cv2.cvtColor(mask_4, cv2.COLOR_BGR2GRAY)

    ret, img1 = video1.read()
    ret, img2 = video2.read()
    ret, img3 = video3.read()
    ret, img4 = video4.read()

    data, voxels_on = table.get_voxels([mask_1, mask_2, mask_3, mask_4])

    cluster_data, _ = find_clusters(voxels_on)
    new_hists = get_histograms([img1, img2, img3, img4], cluster_data, table)

    distances = calculate_distances(histograms, new_hists)
    labels = hungarian_algorithm(distances)
    voxel_data = adjust_labels(cluster_data, labels)
    colors = get_colors(voxel_data)
    return data, colors


# Method to get the camera positions (translation)
def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    v = []  # List to store the coordinates of all the cameras
    for i in range(4):
        cam = i + 1
        file_name = f"data/cam{cam}/config.xml"

        # Obtaining the translation and rotation data
        reader = cv2.FileStorage(file_name, cv2.FileStorage_READ)

        t_vecs = reader.getNode("translation_vectors").mat()
        r_vecs = reader.getNode("rotation_vectors").mat()
        r_vecs, _ = cv2.Rodrigues(r_vecs)

        t_vecs = -np.matrix(r_vecs).T * np.matrix(t_vecs)

        t_vecs = t_vecs.astype(int)
        translation= [(t_vecs[0,0]-500)/20, -t_vecs[2,0]/20, t_vecs[1,0]/20]

        v.append(translation)
    return v, \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


# Method to get the rotation matrices
def get_cam_rotation_matrices():
    cam_rotations = []
    for i in range(4):
        cam = i + 1
        file_name = f"data/cam{cam}/config.xml"

        # Reading the rotation vectors from the config.xml file
        reader = cv2.FileStorage(file_name, cv2.FileStorage_READ)

        r_vecs = reader.getNode("rotation_vectors").mat()

        # Calculating the rotation matrix from the rotation vector
        r_mat = cv2.Rodrigues(r_vecs)

        # Putting the rotation matrix in the upper left corner of a 4 by 4 identity matrix
        I = np.identity(4, dtype=np.float64)
        for i in range(3):
            for j in range(3):
                I[i, j] = r_mat[0][i, j]

        # Correct the rotation (glm treats the rotation matrices differently apparently)
        mat = glm.mat4(I)
        mat = glm.rotate(mat, glm.radians(90), (0, 0, 1))

        cam_rotations.append(mat)
    return cam_rotations

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
