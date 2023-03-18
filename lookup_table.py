# Computer Vision: Assignment 3
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 18-03-2023

import cv2
import numpy as np
from collections import defaultdict


class LookupTable:
    """
    Lookup table class. An object of this class represents a lookup table and has methods for calculating voxels
    that should be turned on given a set of foreground masks.
    """
    def __init__(self, width, depth, height, voxel_size):
        #self.voxel_space = np.zeros((width, depth, height, 4), dtype=bool)
        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size

        # Dictionaries that will serve as lookup tables
        self.lookup_table = defaultdict(list)
        self.voxel2coord = np.ones((width, depth, height, 4, 2), dtype=int)

        # Obtain camera intrinsics and extrinsics per view
        self.cameras = self.configure_cameras()

        # Create the lookup table
        self.create_lookup_table()

    # Function to create lookup table
    def create_lookup_table(self):
        for i, camera in enumerate(self.cameras):

            # Retrieving the camera parameters
            r_vecs = camera.r_vecs
            t_vecs = camera.t_vecs
            mtx = camera.camera_matrix
            dst = camera.distortion_coef

            # Looping over all the voxels
            for x in np.arange(0, self.width):
                for y in np.arange(0, self.depth):
                    for z in np.arange(0, self.height):

                        # Creating an array of the voxel coordinates
                        voxel = np.float32([x * self.voxel_size - 1900, y * self.voxel_size - 700, -z * self.voxel_size])

                        # Calculating the image points of the voxel
                        coordinates, _ = cv2.projectPoints(voxel, r_vecs, t_vecs, mtx, dst)
                        coordinates = tuple(coordinates[0].ravel().astype(int))

                        # Saving the image points in an array with the voxel coordinates as key
                        self.voxel2coord[x, y, z, i] = np.array(coordinates)

                        # Saving the voxel in a dictionary with the view and image points as key
                        self.lookup_table[i + 1, coordinates].append(Voxel(x, y, z))

    # Method for computing the voxels that should be on
    def get_voxels(self, views):
        # Creating a boolean array indicating whether voxels should be on or off
        voxel_space = np.zeros((self.width, self.depth, self.height, 4), dtype=bool)

        # Looping over all image points and views
        for x in np.arange(0, 486):
            for y in np.arange(0, 644):
                for i, view in enumerate(views):
                    if view[x, y] == 255: # Only consider the foreground pixels

                        # When a voxel should be on according to one of the four views, it is marked
                        for voxel in self.lookup_table[i + 1, (y, x)]:
                            voxel_space[voxel.width, voxel.depth, voxel.height, i] = True

        # Turning the voxels on which are visible in all views
        voxels_on = np.all(voxel_space, axis=3)
        voxels_on = np.nonzero(voxels_on)

        # Swapping the y and z coordinates
        data = np.column_stack((voxels_on[0], voxels_on[2], voxels_on[1])).tolist()

        return data, voxels_on

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
    """
    An object of this class represents a voxel and contains the coordinates of that voxel in world coordinates.
    """
    def __init__(self, width, depth, height):
        self.width = width
        self.depth = depth
        self.height = height


class Camera:
    """
    Class to store the camera intrinsics and extrinsics
    """
    def __init__(self, r_vecs, t_vecs, camera_matrix, distortion_coef):
        self.r_vecs = r_vecs
        self.t_vecs = t_vecs
        self.camera_matrix = camera_matrix
        self.distortion_coef = distortion_coef

