import os
import time
from typing import List, Optional, Tuple
import datetime

import numpy as np

# from gello.cameras.camera import CameraDriver
import cv2
import threading
# from gello.zmq_core.camera_node import ZMQServerCamera, ZMQClientCamera



DEFAULT_CAMERA_PORT = 5000

# import run_env.py to get the start pose of manipulator
def read_start_status(filename="start_status.txt"):
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except IOError:
        print("Could not read the status file.")
        return ""

def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


# class RealSenseCamera(CameraDriver):
class RealSenseCamera():
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id: Optional[str] = None, flip: bool = False):
        import pyrealsense2 as rs

        self._device_id = device_id

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(1)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)

        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self._pipeline.start(config)
        self._flip = flip


    def read(self, img_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=10000)  # Increased timeout to 10 seconds
            color_frame = frames.get_color_frame()
            # depth_frame = frames.get_depth_frame()

            # if not color_frame or not depth_frame:
            if not color_frame:
                raise RuntimeError("Failed to get color or depth frame")

            color_image = np.asanyarray(color_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())

            if img_size is None:
                image = color_image[:, :, ::-1]
                # depth = depth_image
            else:
                image = cv2.resize(color_image, img_size)[:, :, ::-1]
                # depth = cv2.resize(depth_image, img_size)

            # rotate 180 degree's because everything is upside down in order to center the camera
            if self._flip:
                image = cv2.rotate(image, cv2.ROTATE_180)
                # depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
            else:
                # depth = depth[:, :, None]
                pass

            # return image, depth
            return image

        except Exception as e:
            print(f"Error reading frame: {e}")
            return None, None
        
def _debug_read(camera, cam_dir ='cam2', save_datastream=False):
    cv2.namedWindow(cam_dir)
    # cv2.namedWindow("depth")
    counter = 0
    if save_datastream and not os.path.exists(f"images/{cam_dir}"):
        os.makedirs(f"images/{cam_dir}")
    # if save_datastream and not os.path.exists("stream"):
        # os.makedirs("stream")
    
    folder_name = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    while True:
        # time.sleep(0.1)
        # image, depth = camera.read()
        image = camera.read()
        
        # if image is None or depth is None:
        if image is None:
            print("Failed to read frame, retrying...")
            continue

        # depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)

        crop_size = (480, 480)
        # shiftx = -35
        shiftx = 0
        # Crop the image
        h, w = image.shape[:2]
        start_x = max(0, w//2 - crop_size[0]//2 + shiftx)
        start_y = max(0, h//2 - crop_size[1]//2)
        end_x = min(w, start_x + crop_size[0])
        end_y = min(h, start_y + crop_size[1])
        image = image[start_y:end_y, start_x:end_x]

        image = cv2.resize(image, (320, 320))
        # image_vis = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)     # visualization - temporary
        cv2.imshow(cam_dir, image[:, :, ::-1])
        # cv2.imshow("depth", depth)

        # save_dir  = ["stream", f"images/{folder_name}"]
        save_dir  =  f"images/{cam_dir}/{folder_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # if key == ord("s"):
        #     cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])

        # Check Status
        start_status = read_start_status()
        print(start_status)

        if save_datastream and start_status == "Reached Start":
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime('%H_%M_%S_%f')[:-3]
            # print(f"Saving image_{timestamp}.png")
            cv2.imwrite(f"{save_dir}/image_{timestamp}.png", image[:, :, ::-1])

        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    # if len(device_ids) == 0:
        # print("No Realsense Camera Found")
    # elif len(device_ids) == 1:
    rs = RealSenseCamera(flip=False, device_id=device_ids[2])
    im = rs.read()
    _debug_read(rs, 'cam2', save_datastream=True)
        
    # elif len(device_ids) == 2:
    #     rs1 = RealSenseCamera(flip=True, device_id=device_ids[0])
    #     rs2 = RealSenseCamera(flip=True, device_id=device_ids[1])

    #     # Start camera servers in separate threads
    #     server1_thread = threading.Thread(target=_debug_read, args=(rs1, True, 'cam2'))
    #     server2_thread = threading.Thread(target=_debug_read, args=(rs2, True, 'cam2'))
    #     server1_thread.start()
    #     server2_thread.start()



    #     server1_thread.join()
    #     server2_thread.join()
    # im, depth = rs.read()
