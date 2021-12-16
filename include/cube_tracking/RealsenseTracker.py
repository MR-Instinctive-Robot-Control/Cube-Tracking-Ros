## Detector for ArUco Markers with Intel RealSense Camera
## Author: zptang (UMass Amherst)

import time
import json
import numpy as np
import cv2
from ArUcoDetector import ArUcoDetector
from Camera import Camera
from TrajectoryTracker import TrajectoryTracker
from ArUcoGenerator import readConfig
import pyrealsense2 as rs


import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5066

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def main():
    dict_to_use, visualize, grey_color, _ = readConfig('./config.json')

    arucoDetector = ArUcoDetector(dict_to_use)
    tracker = TrajectoryTracker()

    camera = Camera()
    camera.startStreaming()


    
    if visualize: start_time = time.time()
    try:
        while True:
            frame = camera.getNextFrame()
            depth_image, color_image = camera.extractImagesFromFrame(frame)


            # Remove unaligned part of the color_image to grey
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
            pose = tracker.updateTrajectory(frame, result)


            # UDP communication protocol
            corners, ids, rejected_img_points = result
            
            if len(corners) > 0: 
                for i in range(0, len(ids)):

                    id_marker = int(ids[i][0])
                    
                    if pose is not None:
                        pose_with_id = np.concatenate([[id_marker], np.squeeze(pose)])
                        pose_string = np.array2string(pose_with_id, formatter={'float_kind':lambda pose_with_id: "%.12f" % pose_with_id})
                        print(pose_string)
            
                        try:
                          sock.sendto( pose_string.encode(), (UDP_IP, UDP_PORT))
                        except Exception as err:
                          print(err)
                    
            
    


            # Show images
            images = np.hstack((color_image, masked_color_image, depth_colormap))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

            if visualize:
                current_time = time.time()
                '''
                print(current_time-start_time)
                if current_time - start_time >= 20:
                    tracker.plotTrajectory()
                    tracker.clear()
                    start_time = current_time
                '''

    finally:
        camera.stopStreaming()



if __name__ == '__main__':
    main()