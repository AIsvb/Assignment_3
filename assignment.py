# Computer Vision: Assignment 3
# Creators: Doyran, M., Gino Kuiper and Sander van Bennekom
# Date: 18-03-2023

import glm
from lookup_table import LookupTable as LT
from clustering_and_matching import *
from matplotlib import pyplot as plt
import numpy as np
import cv2


# Creating all video handlers
foreground1 = cv2.VideoCapture("data/cam1/foreground_cropped.avi")
foreground2 = cv2.VideoCapture("data/cam2/foreground_cropped.avi")
foreground3 = cv2.VideoCapture("data/cam3/foreground_cropped.avi")
foreground4 = cv2.VideoCapture("data/cam4/foreground_cropped.avi")

video1 = cv2.VideoCapture("data/cam1/video_cropped.avi")
video2 = cv2.VideoCapture("data/cam2/video_cropped.avi")
video3 = cv2.VideoCapture("data/cam3/video_cropped.avi")
video4 = cv2.VideoCapture("data/cam4/video_cropped.avi")

foreground_handlers = [foreground1, foreground2, foreground3, foreground4]
video_handlers = [video1, video2, video3, video4]

# Creating variables needed for tracking the path of the persons
n_frames = int(foreground1.get(cv2.CAP_PROP_FRAME_COUNT))
positions = np.empty((n_frames, 4, 2), dtype=int)
frame_no = 0

# Creating the lookup table
x = 3400
y = 4700
z = 2000
dx = 1900
dy = 700

voxel_size = 40
table = LT(int(x/voxel_size), int(y/voxel_size), int(z/voxel_size), voxel_size)

# An array for storing the reference color models
histograms = np.empty((4, 4, 16, 16), dtype=np.float32)

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


# Function that computes the voxel data in the offline phase
def set_voxel_positions():
    global histograms, table, frame_no, positions

    # Read all masks and images
    masks = []
    images = []

    for i, fg_handler in enumerate(foreground_handlers):
        _, mask = fg_handler.read()
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masks.append(mask)

    for j, video_handler in enumerate(video_handlers):
        _, img = video_handler.read()
        images.append(img)

    # Calculate all voxels that should be on according to the foreground masks
    data, voxels_on = table.get_voxels(masks)

    # Cluster the voxels
    cluster_data, centers = find_clusters(voxels_on, 1)

    # Save the cluster centers
    positions[frame_no] = centers
    frame_no += 1

    # Compute the reference color models for the clusters
    histograms = get_histograms(images, cluster_data, table)

    # Get the colors for the voxels
    colors = get_colors(cluster_data)

    return np.column_stack((cluster_data[:, 0], cluster_data[:, 2], cluster_data[:, 1])).tolist(), colors


# Function that computes the voxel data in the online phase
def set_voxel_positions_live():
    global histograms, table, frame_no, positions

    # Read all masks and images
    masks = []
    images = []

    for i, fg_handler in enumerate(foreground_handlers):
        _, mask = fg_handler.read()
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masks.append(mask)

    for j, video_handler in enumerate(video_handlers):
        _, img = video_handler.read()
        images.append(img)

    # Calculate all voxels that should be on according to the foreground masks
    data, voxels_on = table.get_voxels(masks)

    # Cluster the voxels
    cluster_data, centers = find_clusters(voxels_on, 1)

    # Save the cluster centers
    positions[frame_no] = centers
    frame_no += 1

    # Compute the color models for the clusters
    new_hists = get_histograms(images, cluster_data, table)

    # Match the color models with the reference models and adjust the voxels labels accordingly
    distances = calculate_distances(histograms, new_hists)
    labels = hungarian_algorithm(distances)
    voxel_data = adjust_labels(cluster_data, labels)

    # Get the colors for the voxels
    colors = get_colors(voxel_data)

    if True:
        tags = ["blue", "green", "red", "yellow"]
        for i in np.arange(0, 4):
            plt.scatter(positions[0:frame_no, i, 1], positions[0:frame_no, i, 0], c=tags[labels[i]])
        plt.show()

    return np.column_stack((cluster_data[:, 0], cluster_data[:, 2], cluster_data[:, 1])).tolist(), colors


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
        translation= [(t_vecs[0,0] + dx)/voxel_size, -t_vecs[2,0]/voxel_size, (t_vecs[1,0] + dy)/voxel_size]

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