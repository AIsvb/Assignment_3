# Computer Vision: Assignment 2
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 04-03-2023

import cv2
import numpy as np
from collections import defaultdict


class LookupTable:
    def __init__(self, width, height, depth):
        self.voxel_space = np.zeros((width, depth, height, 4), dtype=bool)
        self.width = width
        self.height = height
        self.depth = depth

        # Dictionary that will serve as lookup table
        self.lookup_table = defaultdict(list)
        #self.voxel2coord = np.ones((width, depth, height, 4, 2), dtype=int)

        # Camera intrinsics and extrinsics per view
        self.cameras = self.configure_cameras()

        self.create_lookup_table()

    # Function to create lookup table
    def create_lookup_table(self):
        for i, camera in enumerate(self.cameras):
            r_vecs = camera.r_vecs
            t_vecs = camera.t_vecs
            mtx = camera.camera_matrix
            dst = camera.distortion_coef
            for x in range(self.width):
                for y in range(self.depth):
                    for z in range(self.height):
                        voxel = np.float32([x * 20 - 1500, y * 20 - 500, -z * 20])
                        coordinates, _ = cv2.projectPoints(voxel, r_vecs, t_vecs, mtx, dst)
                        coordinates = tuple(coordinates[0].ravel().astype(int))
                        #print(coordinates)
                        #self.voxel2coord[x, y, z, i] = np.array(coordinates)
                        self.lookup_table[i + 1, coordinates].append(Voxel(x, y, z))
    def get_voxels(self, views):
        for x in range(486):
            for y in range(644):
                for i, view in enumerate(views):
                    if view[x, y] == 255:
                        for voxel in self.lookup_table[i + 1, (y, x)]:
                            self.voxel_space[voxel.width, voxel.depth, voxel.height, i] = True

        voxels_on = np.all(self.voxel_space, axis=3)

        #colors_x = self.voxel2coord[voxels_on, :, 0]
        #colors_y = self.voxel2coord[voxels_on, :, 1]
        #for i in range(4):
         #   colors = images[i][colors_x[:, i].tolist(), colors_y[:, i], :]
        voxels_on = np.nonzero(voxels_on)
        data = np.column_stack((voxels_on[0], voxels_on[2], voxels_on[1])).tolist()
        colors = np.zeros((len(data), 3), dtype=int).tolist()
        return data, colors

    def get_voxels_XOR(self):
        pass

    # Function to set camera intrinsics and extrinsics
    def configure_cameras(self):
        cam1 = self.configure_camera(1)
        cam2 = self.configure_camera(2)
        cam3 = self.configure_camera(3)
        cam4 = self.configure_camera(4)

        return [cam1, cam2, cam3, cam4]

    # Method to load the intrinsics and extrinsics for a given camera
    def configure_camera(self, camera):
        reader = cv2.FileStorage(f"data/cam{camera}/config.xml", cv2.FileStorage_READ)
        t_vecs = reader.getNode("translation_vectors").mat()
        r_vecs = reader.getNode("rotation_vectors").mat()
        camera_matrix = reader.getNode("camera_matrix").mat()
        distortion_coef = reader.getNode("distortion_coefficients").mat()

        return Camera(r_vecs, t_vecs, camera_matrix, distortion_coef)

class Voxel:
    def __init__(self, width, depth, height):
        self.width = width
        self.depth = depth
        self.height = height
        self.color = []

    # Function to set the color of a voxel given 4 input frames
    def set_color(self, frame1, frame2, frame3, frame4):
        color1 = frame1[self.cam_coordinates[0][1], self.cam_coordinates[0][0]]
        color2 = frame2[self.cam_coordinates[1][1], self.cam_coordinates[1][0]]
        color3 = frame3[self.cam_coordinates[2][1], self.cam_coordinates[2][0]]
        color4 = frame4[self.cam_coordinates[3][1], self.cam_coordinates[3][0]]

        b = ((int(color1[0]) + int(color2[0]) + int(color3[0]) + int(color4[0])) / 4) / 255
        g = ((int(color1[1]) + int(color2[1]) + int(color3[1]) + int(color4[1])) / 4) / 255
        r = ((int(color1[2]) + int(color2[2]) + int(color3[2]) + int(color4[2])) / 4) / 255

        self.color = [r, g, b]

# Camera class to store intrinsics and extrinsics
class Camera:
    def __init__(self, r_vecs, t_vecs, camera_matrix, distortion_coef):
        self.r_vecs = r_vecs
        self.t_vecs = t_vecs
        self.camera_matrix = camera_matrix
        self.distortion_coef = distortion_coef

# Masks used for initialization (frame 0 of each view)
mask1 = cv2.imread('data/cam1/voxel.png', cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread('data/cam2/voxel.png', cv2.IMREAD_GRAYSCALE)
mask3 = cv2.imread('data/cam3/voxel.png', cv2.IMREAD_GRAYSCALE)
mask4 = cv2.imread('data/cam4/voxel.png', cv2.IMREAD_GRAYSCALE)
