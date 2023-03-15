# Computer Vision: Assignment 2
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 04-03-2023

import cv2
import numpy as np
from collections import defaultdict


class LookupTable:
    def __init__(self, width, depth, height, voxel_size):
        self.voxel_space = np.zeros((width, depth, height, 4), dtype=bool)
        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size

        # Dictionary that will serve as lookup table
        self.lookup_table = defaultdict(list)
        self.voxel2coord = np.ones((width, depth, height, 4, 2), dtype=int)
        self.v2c = {}

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
            for x in np.arange(0, self.width):
                for y in np.arange(0, self.depth):
                    for z in np.arange(0, self.height):
                        voxel = np.float32([x * self.voxel_size - 1900, y * self.voxel_size - 700, -z * self.voxel_size])
                        coordinates, _ = cv2.projectPoints(voxel, r_vecs, t_vecs, mtx, dst)
                        coordinates = tuple(coordinates[0].ravel().astype(int))
                        self.voxel2coord[x, y, z, i] = np.array(coordinates)
                        cube = Voxel(x, y, z)
                        self.lookup_table[i + 1, coordinates].append(cube)
    def get_voxels(self, views):
        self.voxel_space = np.zeros((self.width, self.depth, self.height, 4), dtype=bool)
        for x in np.arange(0, 486):
            for y in np.arange(0, 644):
                for i, view in enumerate(views):
                    if view[x, y] == 255:
                        for voxel in self.lookup_table[i + 1, (y, x)]:
                            self.voxel_space[voxel.width, voxel.depth, voxel.height, i] = True

        voxels_on = np.all(self.voxel_space, axis=3)

        #colors_x = self.voxel2coord[voxels_on, :, 0]
        #colors_y = self.voxel2coord[voxels_on, :, 1]

        #colors_1 = images[0][colors_x[:, 0].tolist(), colors_y[:, 0], :]
        #colors_2 = images[1][colors_x[:, 1].tolist(), colors_y[:, 1], :]
        #colors_3 = images[2][colors_x[:, 2].tolist(), colors_y[:, 2], :]
        #colors_4 = images[3][colors_x[:, 3].tolist(), colors_y[:, 3], :]



        voxels_on = np.nonzero(voxels_on)
        data = np.column_stack((voxels_on[0], voxels_on[2], voxels_on[1])).tolist()
        colors = np.zeros((len(data), 3), dtype=int).tolist()
        return data, colors

    def get_voxels_XOR(self, views):
        for x in np.arange(0, 486):
            for y in np.arange(0, 644):
                for i, view in enumerate(views):
                    if view[x, y] == 255:
                        for voxel in self.lookup_table[i + 1, (y, x)]:
                            value = self.voxel_space[voxel.width, voxel.depth, voxel.height, i]
                            self.voxel_space[voxel.width, voxel.depth, voxel.height, i] = not value

        voxels_on = np.all(self.voxel_space, axis=3)

        voxels_on = np.nonzero(voxels_on)

        data = np.column_stack((voxels_on[0], voxels_on[2], voxels_on[1])).tolist()
        colors = np.zeros((len(data), 3), dtype=int).tolist()

        return data, colors

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

# Camera class to store intrinsics and extrinsics
class Camera:
    def __init__(self, r_vecs, t_vecs, camera_matrix, distortion_coef):
        self.r_vecs = r_vecs
        self.t_vecs = t_vecs
        self.camera_matrix = camera_matrix
        self.distortion_coef = distortion_coef

