#!/usr/bin/env python3

import sys
import time

import cv2
from cv_bridge import CvBridge
from djitellopy import Tello
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image

class TelloDownwardCameraPublisher(Node):
    """
    Publishes the downward camera feed from the Tello Talent (RoboMaster TT) drone.

    Publishes to:
      - /downward/image_raw (sensor_msgs/Image): ROS image stream from downward camera.
    """

    def __init__(self):
        super().__init__('downward_camera_publisher')

        # Declare parameters
        self.declare_parameter('update_rate', 60.0)
        self.declare_parameter('image_topic', '/downward/image_raw')
        self.declare_parameter('auto_connect', True)
        self.declare_parameter('retry_count', 3)

        # Get parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.image_topic = self.get_parameter('image_topic').value
        self.auto_connect = self.get_parameter('auto_connect').value
        self.retry_count = self.get_parameter('retry_count').value

        # Timer
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)

        # Validate initial parameters
        init_params = [
            Parameter('update_rate', Parameter.Type.DOUBLE, self.update_rate),
            Parameter('image_topic', Parameter.Type.STRING, self.image_topic),
            Parameter('auto_connect', Parameter.Type.BOOL, self.auto_connect),
            Parameter('retry_count', Parameter.Type.INTEGER, self.retry_count),
        ]
        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")

        # Publisher and bridge
        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.bridge = CvBridge()

        # Connect to Tello
        self.tello = Tello()
        if self.auto_connect:
            self._connect_tello()

        # Register parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.crop_start_row = None

        self.get_logger().info("TelloDownwardCameraPublisher Start.")

    def _connect_tello(self):
        success = False
        for attempt in range(1, self.retry_count + 1):
            try:
                self.tello.connect()
                battery = self.tello.get_battery()
                self.get_logger().info(f"Tello connected. Battery: {battery}%.")
                self.tello.set_video_direction(Tello.CAMERA_DOWNWARD)
                self.tello.streamon()
                success = True
                break
            except Exception as e:
                self.get_logger().warn(f"[Attempt {attempt}] Connection failed: {e}")
                time.sleep(1)

        if not success:
            raise RuntimeError("Failed to connect to Tello after multiple attempts.")

    def timer_callback(self):
        frame = self.tello.get_frame_read().frame

        if frame is None or frame.size == 0:
            self.get_logger().warn("Empty or invalid frame.")
            return

        try:
            # Detect crop point only once
            if self.crop_start_row is None:
                green_threshold = 200
                for y in range(frame.shape[0]):
                    row = frame[y, :, :]
                    green_pixels = np.sum((row[:, 0] < 100) & (row[:, 1] > green_threshold) & (row[:, 2] < 100))
                    if green_pixels > frame.shape[1] * 0.5:
                        self.crop_start_row = y
                        break

            # Crop if detected
            clean_frame = frame[:self.crop_start_row, :] if self.crop_start_row is not None else frame

            # Resize to 640x480
            resized = cv2.resize(clean_frame, (640, 480))
            msg = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
            self.image_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        for param in params:
            if param.name == 'update_rate':
                if not isinstance(param.value, (int, float)) or param.value <= 0:
                    return SetParametersResult(successful=False, reason='update_rate must be > 0.')
                self.update_rate = float(param.value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)

            elif param.name == 'image_topic':
                if not isinstance(param.value, str) or len(param.value) == 0:
                    return SetParametersResult(successful=False, reason='image_topic must be a non-empty string.')
                self.image_topic = param.value
                self.image_pub = self.create_publisher(Image, self.image_topic, 10)

            elif param.name == 'auto_connect':
                if not isinstance(param.value, bool):
                    return SetParametersResult(successful=False, reason='auto_connect must be true/false.')
                self.auto_connect = param.value

            elif param.name == 'retry_count':
                if not isinstance(param.value, int) or param.value < 0:
                    return SetParametersResult(successful=False, reason='retry_count must be non-negative.')
                self.retry_count = param.value

        return SetParametersResult(successful=True)

    def destroy_node(self):
        try:
            self.tello.streamoff()
        except Exception as e:
            self.get_logger().warn(f"Error while stopping stream: {e}.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TelloDownwardCameraPublisher()
    except Exception as e:
        print(f"[FATAL] TelloDownwardCameraPublisher failed to initialize: {e}.", file=sys.stderr)
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted with Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()