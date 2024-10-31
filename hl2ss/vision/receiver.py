

from pynput import keyboard

import multiprocessing as mp
import numpy as np
import cv2
import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import torch
import sys
import os
from PIL import Image
import threading

from real_time import detect, utils

from scene.env import *


# Settings --------------------------------------------------------------------

# HoloLens address
host = '169.254.174.24'

# Calibration path (must exist but can be empty)
calibration_path = "../calibration"

# Camera parameters
pv_width = 640
pv_height = 360
pv_framerate = 30

# Buffer length in seconds
buffer_length = 10

# Maximum depth, points beyond are removed
max_sensor_depth = 10

device = 'cpu' #'cuda:0'
score_thr = 0.3
wait_time = 1

visualize = True

# For recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('testvid.mp4', fourcc, 20.0,(pv_height, pv_width))


# Init and Terminate ---------------------------------------------------------

def initiate_receiver():

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO,shared=True)

    # Start PV and RM Depth Long Throw streams --------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, buffer_length * pv_framerate)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, buffer_length * hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    manager = mp.Manager()
    consumer = hl2ss_mp.consumer()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, None)
    sink_pv.get_attach_response()
    sink_depth.get_attach_response()

    return sink_pv, sink_depth, producer

def terminate_receiver(sink_pv, sink_depth, producer):

    # Stop PV and RM Depth Long Throw streams ---------------------------------
    sink_pv.detach()
    sink_depth.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)


# CV Stuff --------------------------------------------------------------------
def update_visualization(frame,vis_inference_frame, points, vis, main_pcd, first_geometry):

    cv2.imshow("Object Detection", vis_inference_frame)
    cv2.imshow("RGB", frame)  
    #out.write(frame)

    main_pcd.points = o3d.utility.Vector3dVector(points)

    if (first_geometry):
        vis.add_geometry(main_pcd)
        first_geometry = False
    else:
        vis.update_geometry(main_pcd)

    vis.poll_events()
    vis.update_renderer()

def post_process_poses(depth, poses: List[utils.Object], xy1, depth_to_world):

    """ Transform poses into global frame"""
    # Get list of center pixels for each detected object
    centers = utils.get_centers(poses)

    # Transform to world CF.
    points = hl2ss_3dcv.rm_depth_to_points(depth, xy1)
    points = hl2ss_3dcv.transform(points, depth_to_world)

    set = []
    for cx,cy in centers:
        set.append(hl2ss_3dcv.block_to_list(points[cy,cx])[0])

    # Generate point cloud
    point_cloud = create_point_cloud(set)

    return point_cloud

def frame_processing(data_pv, data_depth, pv_intrinsics, pv_extrinsics, depth_settings):
        
    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Preprocess frames ---------------------------------------------------
    frame = data_pv.payload.image
    sensor_depth = hl2ss_3dcv.rm_depth_normalize(data_depth.payload.depth, depth_settings['scale'])
    sensor_depth[sensor_depth > max_sensor_depth] = 0

    # Build pointcloud ----------------------------------------------------
    points = hl2ss_3dcv.rm_depth_to_points(depth_settings['xy1'],sensor_depth)
    world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
    depth_to_world = hl2ss_3dcv.camera_to_rignode(depth_settings['calibration_depth'].extrinsics) @ hl2ss_3dcv.reference_to_world(data_depth.pose)
    world_points = hl2ss_3dcv.transform(points, depth_to_world)
    pixels = hl2ss_3dcv.project(world_points, world_to_pv_image)

    # Modify depth range for visuals -----------------------------------------------
    vis_depth = cv2.normalize(sensor_depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    vis_depth_frame = cv2.applyColorMap(vis_depth, colormap=cv2.COLORMAP_INFERNO)

    # Object Detection -----------------------------------------------------
    _ , boxes = detect.get_object_detection_frame(frame)
    mapped_boxes = detect.remap_bounding_boxes(boxes,frame,pixels)
    poses = detect.estimate_box_poses(sensor_depth,mapped_boxes)
    vis_depth_frame = detect.draw_depth_boxes(vis_depth_frame,mapped_boxes,poses)
    frame = detect.draw_depth_boxes(frame,boxes,poses)

    # Get world poses for detected objects
    point_cloud = []
    if len(poses) > 0:
        point_cloud = post_process_poses(sensor_depth,poses,depth_settings['xy1'],depth_to_world)

    # Update Environement
    return frame, vis_depth_frame, points, point_cloud

if __name__ == "__main__":

    sink_pv, sink_depth, producer = initiate_receiver()


    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)
    uv2xy = calibration_lt.uv2xy
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

    # Pack all depth settings together for clarity
    depth_settings = {
        'xy1': xy1,
        'scale': scale,
        'calibration_depth': calibration_lt
    }

    found = False
    while not found:

        # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
        _, data_depth = sink_depth.get_most_recent_frame()
        if ((data_depth is None) or (not hl2ss.is_valid_pose(data_depth.pose))):
            continue

        _, data_pv = sink_pv.get_nearest(data_depth.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue


        if data_pv.payload.image is not None:
            found = True
            frame ,vis_inference_frame, points, point_cloud = frame_processing(data_pv,data_depth, pv_intrinsics, pv_extrinsics, depth_settings)

            cv2.imshow("RGB",frame)
            cv2.imshow("DEPTH",vis_inference_frame)
            cv2.waitKey(1000)

    terminate_receiver(sink_pv,sink_depth,producer)






# Older loop code -------------------------------------------------------------


def real_time_loop():
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.space
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO,shared=True)

    # Create visualizers ------------------------------------------------------

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    main_pcd = o3d.geometry.PointCloud()
    first_geometry = True

    # Start PV and RM Depth Long Throw streams --------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, buffer_length * pv_framerate)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, buffer_length * hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    manager = mp.Manager()
    consumer = hl2ss_mp.consumer()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, None)
    sink_pv.get_attach_response()
    sink_depth.get_attach_response()


    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)
    uv2xy = calibration_lt.uv2xy
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    depth_undistort_map = calibration_lt.undistort_map

    # Pack all depth settings together for clarity
    depth_settings = {
        'xy1': xy1,
        'scale': scale,
        'depth_undistort_map': depth_undistort_map
    }

    # Main loop ---------------------------------------------------------------
    while (enable):
        # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
        _, data_depth = sink_depth.get_most_recent_frame()
        if ((data_depth is None) or (not hl2ss.is_valid_pose(data_depth.pose))):
            continue

        _, data_pv = sink_pv.get_nearest(data_depth.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue


        if data_pv.payload.image is not None:
            frame ,vis_inference_frame, points = frame_processing(data_pv,data_depth, pv_intrinsics, pv_extrinsics, depth_settings)

            if visualize:
                update_visualization(frame, vis_inference_frame, points, vis, main_pcd, first_geometry)


    # Stop PV and RM Depth Long Throw streams ---------------------------------
    sink_pv.detach()
    sink_depth.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()
