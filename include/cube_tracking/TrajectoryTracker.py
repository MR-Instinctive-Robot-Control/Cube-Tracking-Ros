## Detector for ArUco Markers with Intel RealSense Camera
## Author: zptang (UMass Amherst)

import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

class TrajectoryTracker:

    def __init__(self):
        self.trajectory = dict()


    def clear(self):
        self.trajectory = dict()



    def updateTrajectory(self, aligned_frame, detectorResult, num_cubes):
        corners, ids, rejected = detectorResult

        timestamp = aligned_frame.get_timestamp()
        depth_frame = aligned_frame.get_depth_frame()
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        corners = np.array(corners)
        ids = ids.reshape(-1)

        #######  OBJECT 1  #######
        object_1_faces = np.sum(np.where(ids<=6,1,0))                  # Number of faces detected of object 1
        corners_object_1 = corners[np.argwhere(ids<=6)]                # Corners of each detected face of object 1
        object_1_id = ids[np.argwhere(ids<=6)]                         # ID of each detected face in object 1


        #######  OBJECT 2  #######
        cond_obj2 = ((ids>6) & (ids<=12))                          
        object_2_faces = np.sum(np.where(cond_obj2, 1, 0))             # Number of faces detected of object 2      
        corners_object_2 = corners[np.argwhere(cond_obj2)]             # Corners of each detected face of object 2
        object_2_id = ids[np.argwhere(cond_obj2)]                      # ID of each detected face in object 2


        #######  OBJECT 3  #######
        cond_obj3 = ((ids>12) & (ids<=18))                        
        object_3_faces = np.sum(np.where(cond_obj3, 1, 0))             # Number of faces detected of object 3
        corners_object_3 = corners[np.argwhere(cond_obj3)]             # Corners of each detected face of object 3
        object_3_id = ids[np.argwhere(cond_obj3)]                      # ID of each detected face in object 3


        #######  OBJECT 4  #######
        cond_obj4 = ((ids>18) & (ids<=24))
        object_4_faces = np.sum(np.where(cond_obj4, 1, 0))             # Number of faces detected of object 4
        corners_object_4 = corners[np.argwhere(cond_obj4)]             # Corners of each detected face of object 4
        object_4_id = ids[np.argwhere(cond_obj4)]                      # ID of each detected face in object 4


        detected_faces = np.array([object_1_faces, object_2_faces, object_3_faces, object_4_faces])                # Number of faces for each object
        detected_objects = np.sum(np.where(detected_faces>0,1,0))                                                  # Total number of objects detected


        new_corners = [corners_object_1, corners_object_2, corners_object_3, corners_object_4]
        new_ids = [object_1_id, object_2_id, object_3_id, object_4_id]


        positions_list = []
        feasible_ids = []

        # We enter the loop if at least one object is detected
        if detected_objects > 0:

            # We loop over the maximum number of cubes detectable 
            for j in range(num_cubes):
                positions = []

                # For each object we loop over all its detected faces
                for i in range(0, detected_faces[j]):

		            # For each detected marker we extract its corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
                    corner = new_corners[j][i].reshape((4, 2))

                    (topLeft, topRight, bottomRight, bottomLeft) = corner

		            # We convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                
                    position = self._getCoordinate(depth_intrinsics, depth_frame, (topRight, bottomRight, bottomLeft, topLeft))    
     
                    positions.append(position)
                    feasible_ids.append(new_ids[j][i])

             
                positions_list.append(positions)
            
            return positions_list, feasible_ids



    def updateTrajectory_calibration(self, aligned_frame, detectorResult, calibration_marker_1, calibration_marker_2):
        corners, ids, rejected = detectorResult

        timestamp = aligned_frame.get_timestamp()
        depth_frame = aligned_frame.get_depth_frame()
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics


        if len(corners) > 0:
	        # flatten the ArUco IDs list
            ids = ids.flatten()

	        # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):

		        # extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

		        # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                
                position = self._getCoordinate(depth_intrinsics, depth_frame, (topRight, bottomRight, bottomLeft, topLeft))    

                
                if position is not None and ((markerID == calibration_marker_1) or (markerID == calibration_marker_2)):
                    self._add(timestamp, markerID, position)
                    return position, markerID

            return 0
                
 
    
    def _add(self, timestamp, id, coord):
        if id not in self.trajectory.keys():
            self.trajectory[id] = list()
        x, y, z = coord
        self.trajectory[id].append((timestamp, x, y, z))
        #print(self.trajectory[0][-10:], '\n')


    def _getCoordinate(self, depth_intrinsics, depth_frame, markerPos):

        # for simplicity, just use the center point to extract the 3D coordinate
        topRight, bottomRight, bottomLeft, topLeft = markerPos
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)

        depth = depth_frame.get_distance(cX, cY)

        if depth > 0:
            return rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cX, cY], depth)
        else:
            return None


    def plotTrajectory(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for id in self.trajectory.keys():
            x_line = [x for t, x, y, z in self.trajectory[id]]
            y_line = [y for t, x, y, z in self.trajectory[id]]
            z_line = [z for t, x, y, z in self.trajectory[id]]
            ax.scatter3D(x_line, y_line, z_line)

        plt.show()



if __name__ == '__main__':
    import time
    import cv2
    from ArUcoDetector import ArUcoDetector
    from Camera import Camera

    visualize = True
    dict_to_use = 'DICT_5X5_50'
    arucoDetector = ArUcoDetector(dict_to_use)

    tracker = TrajectoryTracker()

    camera = Camera()
    camera.startStreaming()
    
    start_time = time.time()
    try:
        while True:
            frame = camera.getNextFrame()
            depth_image, color_image = camera.extractImagesFromFrame(frame)

            # Remove unaligned part of the color_image to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            masked_color_image = np.where(depth_image_3d <= 0, grey_color, color_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Detect markers and draw them on the images
            result = arucoDetector.detect(color_image)
            color_image = ArUcoDetector.getImageWithMarkers(color_image, result)
            masked_color_image = ArUcoDetector.getImageWithMarkers(masked_color_image, result)
            depth_colormap = ArUcoDetector.getImageWithMarkers(depth_colormap, result)

            # Update trajectory
            tracker.updateTrajectory(frame, result)

            # Show images
            images = np.hstack((color_image, masked_color_image, depth_colormap))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

            current_time = time.time()
            print(current_time-start_time)
            if current_time - start_time >= 20:
                if visualize:
                    tracker.plotTrajectory()
                tracker.clear()
                start_time = current_time
    finally:
        camera.stopStreaming()
