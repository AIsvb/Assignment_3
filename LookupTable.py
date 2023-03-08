# Computer Vision: Assignment 2
# Creators: Gino Kuiper and Sander van Bennekom
# Date: 04-03-2023

import cv2
import numpy as np
import PIL


class LookupTable:
    def __init__(self, width, height, depth, mask1, mask2, mask3, mask4):
        self.width = width
        self.height = height
        self.depth = depth

        # List of voxels
        self.voxels = []

        # Dictionary that will serve as lookup table
        self.lookup_table = {}

        # Camera intrinsics and extrinsics per view
        self.cam1 = self.cam2 = self.cam3 = self.cam4 = 0
        self.configure_cameras()

        # Masks used for initialization of voxel list and lookup table
        self.masks = [mask1, mask2, mask3, mask4]

        # Colored frames to color the voxels
        self.colored_frame1 = cv2.imread('data/cam1/color.png')
        self.colored_frame2 = cv2.imread('data/cam2/color.png')
        self.colored_frame3 = cv2.imread('data/cam3/color.png')
        self.colored_frame4 = cv2.imread('data/cam4/color.png')


    # Function to fill voxel list
    def create_voxels(self):
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    self.voxels.append(Voxel(x * 20 - 1500, y * 20 - 500, -z * 20, self.cam1, self.cam2, self.cam3, self.cam4))

    # Function to create lookup table
    def create_lookup_table(self):
        for v in self.voxels:
            for i in range(4):
                try:
                    self.lookup_table[i + 1, (v.cam_coordinates[i][0], v.cam_coordinates[i][1])].append(v)
                except (KeyError):
                    self.lookup_table[i + 1, (v.cam_coordinates[i][0], v.cam_coordinates[i][1])] = [v]

    # Function to switch voxels on/off
    def get_voxels(self, mask1, mask2, mask3, mask4):
        list = []
        for x in range(644):
            for y in range(486):
                    if mask1[y,x][0] == 255:
                        try:
                            for v in self.lookup_table[1, (x,y)]:
                                v.show1 = 1
                        except (KeyError):
                            0
                    if mask2[y,x][0] == 255:
                        try:
                            for v in self.lookup_table[2, (x,y)]:
                                v.show2 = 1
                        except (KeyError):
                            0
                    if mask3[y,x][0] == 255:
                        try:
                            for v in self.lookup_table[3, (x,y)]:
                                v.show3 = 1
                        except (KeyError):
                            0
                    if mask4[y,x][0] == 255:
                        try:
                            for v in self.lookup_table[4, (x,y)]:
                                v.show4 = 1
                        except (KeyError):
                            0

        # Loop over all voxels, if 'on' in all 4 views, color it and add it to the list
        for v in self.voxels:
            if v.show1 + v.show2 + v.show3 + v.show4 == 4:
                v.set_color(self.colored_frame1, self.colored_frame2, self.colored_frame3, self.colored_frame4)
                list.append(v)

        return list

    # Function to get changed voxels between two frames
    def get_voxels_XOR(self, list1, list2, list3, list4, complete_list):
        # Already existing list with voxels that are turned 'on'
        voxel_list = complete_list

        # List to store voxels that have changed
        changed_voxels = []

        # For every XOR-frame, loop over all pixels. If it's white, switch the 'show' property of the corresponding voxels
        # and add it to the list of changed voxels.
        for x in range(644):
            for y in range(486):
                    if list1[y,x][0] == 255:
                        try:
                            for v in self.lookup_table[1, (x, y)]:
                                if v.show1 == 1:
                                    v.show1 = 0
                                    changed_voxels.append(v)
                                else:
                                    v.show1 = 1
                                    changed_voxels.append(v)
                        except (KeyError):
                            0
                    if list2[y,x][0] == 255:
                        try:
                            for v in self.lookup_table[2, (x, y)]:
                                if v.show2 == 1:
                                    v.show2 = 0
                                    changed_voxels.append(v)
                                else:
                                    v.show2 = 1
                                    changed_voxels.append(v)
                        except (KeyError):
                            0
                    if list3[y,x][0] == 255:
                        try:
                            for v in self.lookup_table[3, (x, y)]:
                                if v.show3 == 1:
                                    v.show3 = 0
                                    changed_voxels.append(v)
                                else:
                                    v.show3 = 1
                                    changed_voxels.append(v)
                        except (KeyError):
                            0
                    if list4[y,x][0] == 255:
                        try:
                            for v in self.lookup_table[4, (x, y)]:
                                if v.show4 == 1:
                                    v.show4 = 0
                                    changed_voxels.append(v)
                                else:
                                    v.show4 = 1
                                    changed_voxels.append(v)
                        except (KeyError):
                            0

        # Loop over all changed voxels and turn them 'on' or 'off' by adding or removing them from the final list
        for v in changed_voxels:
            if v.show1 + v.show2 + v.show3 + v.show4 == 4:
                            v.set_color(self.colored_frame1, self.colored_frame2, self.colored_frame3,
                                        self.colored_frame4)
                            voxel_list.append(v)
            else:
                if v in voxel_list:
                    voxel_list.remove(v)
        return voxel_list

    # Function to set camera intrinsics and extrinsics
    def configure_cameras(self):
        self.cam1 = self.configure_camera(1)
        self.cam2 = self.configure_camera(2)
        self.cam3 = self.configure_camera(3)
        self.cam4 = self.configure_camera(4)

    # Method to load the intrinsics and extrinsics for a given camera
    def configure_camera(self, camera):
        reader = cv2.FileStorage(f"data/cam{camera}/config.xml", cv2.FileStorage_READ)
        t_vecs = reader.getNode("translation_vectors").mat()
        r_vecs = reader.getNode("rotation_vectors").mat()
        camera_matrix = reader.getNode("camera_matrix").mat()
        distortion_coef = reader.getNode("distortion_coefficients").mat()

        return Camera(r_vecs, t_vecs, camera_matrix, distortion_coef)

class Voxel:
    def __init__(self, width, height, depth, cam1, cam2, cam3, cam4):
        self.voxel_coordinates = (width, height, depth)
        self.color = []

        # Per view, determine the image points and store them in cam_coordinates
        self.cam1_coordinates, _ = cv2.projectPoints(self.voxel_coordinates, cam1.r_vecs, cam1.t_vecs, cam1.camera_matrix, cam1.distortion_coef)
        self.cam2_coordinates, _ = cv2.projectPoints(self.voxel_coordinates, cam2.r_vecs, cam2.t_vecs, cam2.camera_matrix, cam2.distortion_coef)
        self.cam3_coordinates, _ = cv2.projectPoints(self.voxel_coordinates, cam3.r_vecs, cam3.t_vecs, cam3.camera_matrix, cam3.distortion_coef)
        self.cam4_coordinates, _ = cv2.projectPoints(self.voxel_coordinates, cam4.r_vecs, cam4.t_vecs, cam4.camera_matrix, cam4.distortion_coef)

        self.cam_coordinates = [(int(self.cam1_coordinates[0][0][0]),int(self.cam1_coordinates[0][0][1])), (int(self.cam2_coordinates[0][0][0]),int(self.cam2_coordinates[0][0][1])),
                                (int(self.cam3_coordinates[0][0][0]), int(self.cam3_coordinates[0][0][1])), (int(self.cam4_coordinates[0][0][0]),int(self.cam4_coordinates[0][0][1]))]

        # Integer per view whether the voxel is on the foreground, 0 = off, 1 = on.
        self.show1 = 0
        self.show2 = 0
        self.show3 = 0
        self.show4 = 0

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