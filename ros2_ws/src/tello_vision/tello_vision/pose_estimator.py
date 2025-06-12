#!/usr/bin/env python3

import sys
import numpy as np
import cv2
import threading
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped, PoseStamped
from std_msgs.msg import Header

from tf2_ros import TransformBroadcaster

from tello_utils.matrix_helpers import (
    reverse_xyz_to_zyx_4x4,
    extract_euler_zyx,
    Rx, Ry, Rz,
    vecs_to_matrix,
    Rx180
)

class CharuCoPoseEstimator(Node):
    """
    Estimates drone pose from CharuCo board detection using OpenCV.
    Publishes the pose for controller consumption and TF for visualization in RViz.
    
    The drone/camera pose is estimated relative to a stationary CharuCo board,
    which serves as the reference frame.
    """
    def __init__(self):
        super().__init__('pose_estimator')
        
        # Declare parameters
        self.declare_parameter('update_rate', 30.0)                    # Hz
        self.declare_parameter('image_topic', '/downward/image_raw')   # Image subscription topic
        
        # Camera calibration parameters
        self.declare_parameter('camera_matrix', [226.88179248, 0.0, 160.67144106,
                                                0.0, 227.22020997, 117.68560542,
                                                0.0, 0.0, 1.0])
        self.declare_parameter('distortion_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
        
        # CharuCo board parameters
        self.declare_parameter('board_squares_x', 9)
        self.declare_parameter('board_squares_y', 24)
        self.declare_parameter('square_length', 0.1)                   # m
        self.declare_parameter('marker_length', 0.08)                  # m
        self.declare_parameter('aruco_dict', 'DICT_4X4_250')
        self.declare_parameter('center_board_origin', True)
        self.declare_parameter('min_corners_detected', 6)              # Minimum corners for valid pose
        
        # TF frame names
        self.declare_parameter('world_frame', 'world')                 # For RViz visualization
        self.declare_parameter('board_frame', 'charuco_board')         # Control reference frame
        self.declare_parameter('drone_frame', 'drone')                 # Drone/camera frame
        
        # Visualization options
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('publish_tf', True)
        
        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.image_topic = self.get_parameter('image_topic').value
        
        camera_matrix_flat = self.get_parameter('camera_matrix').value
        self.camera_matrix = np.array(camera_matrix_flat).reshape(3, 3)
        self.dist_coeffs = np.array(self.get_parameter('distortion_coeffs').value)
        
        self.board_squares_x = self.get_parameter('board_squares_x').value
        self.board_squares_y = self.get_parameter('board_squares_y').value
        self.square_length = self.get_parameter('square_length').value
        self.marker_length = self.get_parameter('marker_length').value
        self.aruco_dict_name = self.get_parameter('aruco_dict').value
        self.center_board_origin = self.get_parameter('center_board_origin').value
        self.min_corners_detected = self.get_parameter('min_corners_detected').value
        
        self.world_frame = self.get_parameter('world_frame').value
        self.board_frame = self.get_parameter('board_frame').value
        self.drone_frame = self.get_parameter('drone_frame').value
        
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.publish_tf = self.get_parameter('publish_tf').value

        # Timer for periodic processing
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on-set-parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Immediately validate the initial values
        init_params = [
            Parameter('update_rate', Parameter.Type.DOUBLE, self.update_rate),
            Parameter('image_topic', Parameter.Type.STRING, self.image_topic),
            Parameter('camera_matrix', Parameter.Type.DOUBLE_ARRAY, camera_matrix_flat),
            Parameter('distortion_coeffs', Parameter.Type.DOUBLE_ARRAY, self.dist_coeffs.tolist()),
            Parameter('board_squares_x', Parameter.Type.INTEGER, self.board_squares_x),
            Parameter('board_squares_y', Parameter.Type.INTEGER, self.board_squares_y),
            Parameter('square_length', Parameter.Type.DOUBLE, self.square_length),
            Parameter('marker_length', Parameter.Type.DOUBLE, self.marker_length),
            Parameter('aruco_dict', Parameter.Type.STRING, self.aruco_dict_name),
            Parameter('center_board_origin', Parameter.Type.BOOL, self.center_board_origin),
            Parameter('min_corners_detected', Parameter.Type.INTEGER, self.min_corners_detected),
            Parameter('world_frame', Parameter.Type.STRING, self.world_frame),
            Parameter('board_frame', Parameter.Type.STRING, self.board_frame),
            Parameter('drone_frame', Parameter.Type.STRING, self.drone_frame),
            Parameter('publish_debug_image', Parameter.Type.BOOL, self.publish_debug_image),
            Parameter('publish_tf', Parameter.Type.BOOL, self.publish_tf),
        ]

        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize variables
        self.bridge = CvBridge()
        self.current_image = None
        self.image_header = None
        self.frame_count = 0
        
        # Create CharuCo board and detector
        self.create_charuco_board()
        
        # Threading for non-blocking pose estimation
        self._lock = threading.Lock()
        self._thread_running = False
        self._last_drone_pose = None
        
        # Create subscriber for camera image
        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos.qos_profile_sensor_data
        )
        
        # Create publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            'drone/pose',
            10
        )
        
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(
                Image,
                'drone/debug_image',
                10
            )
        
        # Create TF broadcaster
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info("CharuCoPoseEstimator Start.")

    def create_charuco_board(self):
        """Create CharuCo board and detector."""
        # Get ArUco dictionary
        dict_mapping = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        }
        
        if self.aruco_dict_name not in dict_mapping:
            raise RuntimeError(f"Unknown ArUco dictionary: {self.aruco_dict_name}")
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_mapping[self.aruco_dict_name])
        
        # Create CharuCo board
        self.board = cv2.aruco.CharucoBoard(
            size=(self.board_squares_x, self.board_squares_y),
            squareLength=self.square_length,
            markerLength=self.marker_length,
            dictionary=aruco_dict
        )
        
        # Create detector
        self.detector = cv2.aruco.CharucoDetector(self.board)
        
        self.get_logger().info(
            f"Created CharuCo board: {self.board_squares_x}x{self.board_squares_y}, "
            f"square={self.square_length}m, marker={self.marker_length}m"
        )

    def image_callback(self, msg):
        """Store the latest camera image."""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_header = msg.header
            self.frame_count += 1
            
            # Log first frame info
            if self.frame_count == 1:
                self.get_logger().info(f"First image received! Size: {msg.width}x{msg.height}, Encoding: {msg.encoding}.")
                
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def timer_callback(self):
        """Main processing loop - estimate and publish drone pose."""
        if self.current_image is None:
            return
        
        # Get drone pose (non-blocking)
        result = self.get_drone_transform_nb(self.current_image.copy())
        
        if result is not None:
            drone_T, charuco_corners, charuco_ids = result
            
            # Publish pose for controller
            self.publish_drone_pose(drone_T)
            
            # Publish TFs for visualization
            if self.publish_tf:
                self.publish_tfs(drone_T)
            
            # Publish debug image
            if self.publish_debug_image:
                self.publish_debug_image_msg(self.current_image.copy(), charuco_corners, charuco_ids)
            
            # Log detection with throttle
            self.get_logger().info(
                f"Detected {len(charuco_ids)} CharuCo corners, publishing pose",
                throttle_duration_sec=2.0
            )
        else:
            # Log no detection with throttle
            self.get_logger().info(
                "No CharuCo board detected",
                throttle_duration_sec=5.0
            )

    def get_drone_transform_nb(self, frame):
        """
        Non-blocking drone transform estimation.
        Returns the most recent transform while computing new one in background.
        """
        with self._lock:
            # If no thread is active, start one
            if not self._thread_running:
                self._thread_running = True

                def worker(f):
                    try:
                        new_result = self.estimate_drone_pose(f)
                        with self._lock:
                            self._last_drone_pose = new_result
                    except Exception as e:
                        self.get_logger().error(f"Pose estimation error: {e}")
                    finally:
                        with self._lock:
                            self._thread_running = False

                thread = threading.Thread(target=worker, args=(frame,))
                thread.daemon = True
                thread.start()

            # Return last known transform
            return self._last_drone_pose

    def estimate_drone_pose(self, frame):
        """Estimate drone pose from CharuCo board detection."""
        # Detect CharuCo board
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(frame)
        
        if charuco_ids is None or len(charuco_ids) < self.min_corners_detected:
            return None
        
        # Get board pose
        obj_pts, img_pts = self.board.matchImagePoints(charuco_corners, charuco_ids)
        
        if self.center_board_origin:
            # Center the board origin
            center_board = np.array([
                ((self.board_squares_x - 1) * self.square_length / 2.0) + self.square_length / 2.0,
                (self.board_squares_y - 1) * self.square_length / 2.0 + self.square_length / 2.0,
                0.0
            ], dtype=np.float64)
            obj_pts = obj_pts - center_board
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # Convert to transformation matrix
        board_T = vecs_to_matrix(rvec.flatten(), tvec.flatten())
        
        # Get camera pose relative to board
        cam_T = np.linalg.inv(board_T)
        
        # Fix camera transform
        cam_T = self.fix_camera_transform(cam_T)
        
        # For drone, we use camera transform directly
        drone_T = cam_T
        
        return drone_T, charuco_corners, charuco_ids

    def fix_camera_transform(self, cam_T):
        """
        Fix camera transform to match coordinate system conventions.
        """
        # Create 4x4 version of Rx180 for this operation
        Rx180_4x4 = np.eye(4)
        Rx180_4x4[1, 1] = -1
        Rx180_4x4[2, 2] = -1
        
        # Apply X-axis 180° rotation
        cam_T = Rx180_4x4 @ cam_T @ Rx180_4x4

        # Extract rotation and translation
        R = cam_T[:3, :3].copy()
        t = cam_T[:3, 3].copy()

        # Mirror Y reflection for rotation matrix only (3x3)
        mirror_y_3 = np.diag([1, -1, 1])
        R = mirror_y_3 @ R @ mirror_y_3
        
        # Put the corrected rotation back into cam_T
        cam_T[:3, :3] = R
        cam_T[:3, 3] = t

        # Fix X/Z ordering
        cam_T = reverse_xyz_to_zyx_4x4(cam_T)

        # Extract and fix Y rotation
        R_rev = cam_T[:3, :3].copy()
        t_rev = cam_T[:3, 3].copy()

        alpha, beta, gamma = extract_euler_zyx(R_rev)
        beta = -beta

        # Rebuild rotation (these functions return 3x3 matrices)
        R_fixed = Rx(alpha) @ Ry(beta) @ Rz(gamma)

        # Final transform
        cam_T[:3, :3] = R_fixed
        cam_T[:3, 3] = t_rev

        return cam_T

    def publish_drone_pose(self, drone_T):
        """Publish drone pose as PoseStamped message for controller."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.board_frame
        
        # Extract position
        pose_msg.pose.position.x = float(drone_T[0, 3])
        pose_msg.pose.position.y = float(drone_T[1, 3])
        pose_msg.pose.position.z = float(drone_T[2, 3])
        
        # Extract quaternion from rotation matrix
        R = drone_T[:3, :3]
        quat = self.rotation_matrix_to_quaternion(R)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.pose_pub.publish(pose_msg)

    def publish_tfs(self, drone_T):
        """Publish TF transforms for RViz visualization."""
        now = self.get_clock().now().to_msg()
        
        # Static transform: world -> board
        board_tf = TransformStamped()
        board_tf.header.stamp = now
        board_tf.header.frame_id = self.world_frame
        board_tf.child_frame_id = self.board_frame
        board_tf.transform.translation.x = 0.0
        board_tf.transform.translation.y = 0.0
        board_tf.transform.translation.z = 0.0
        board_tf.transform.rotation.x = 0.0
        board_tf.transform.rotation.y = 0.0
        board_tf.transform.rotation.z = 0.0
        board_tf.transform.rotation.w = 1.0
        
        # Dynamic transform: board -> drone
        drone_tf = TransformStamped()
        drone_tf.header.stamp = now
        drone_tf.header.frame_id = self.board_frame
        drone_tf.child_frame_id = self.drone_frame
        
        drone_tf.transform.translation.x = float(drone_T[0, 3])
        drone_tf.transform.translation.y = float(drone_T[1, 3])
        drone_tf.transform.translation.z = float(drone_T[2, 3])
        
        R = drone_T[:3, :3]
        quat = self.rotation_matrix_to_quaternion(R)
        drone_tf.transform.rotation.x = quat[0]
        drone_tf.transform.rotation.y = quat[1]
        drone_tf.transform.rotation.z = quat[2]
        drone_tf.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform([board_tf, drone_tf])

    def publish_debug_image_msg(self, image, corners, ids):
        """Publish visualization image with detections drawn."""
        # Draw detected corners
        cv2.aruco.drawDetectedCornersCharuco(
            image, corners, ids, 
            cornerColor=(0, 255, 255)
        )
        
        # Add text overlay
        cv2.putText(
            image, 
            f"Detected: {len(ids)} corners", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            debug_msg.header = self.image_header if self.image_header else Header()
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_image_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Debug image publish error: {e}")

    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x,y,z,w]."""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])

    def parameter_callback(self, params):
        """Handle parameter updates."""
        for param in params:
            name = param.name
            value = param.value

            if name == 'update_rate':
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason="update_rate must be > 0.")
                self.update_rate = float(value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
                self.get_logger().info(f"Updated update_rate: {self.update_rate} Hz.")

            elif name == 'image_topic':
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason="image_topic must be a string.")
                self.image_topic = value
                self.get_logger().info(f"Updated image_topic: {value}")

            elif name == 'camera_matrix':
                if not isinstance(value, list) or len(value) != 9:
                    return SetParametersResult(successful=False, reason="camera_matrix must be a list of 9 values.")
                self.camera_matrix = np.array(value).reshape(3, 3)
                self.get_logger().info("Updated camera_matrix")

            elif name == 'distortion_coeffs':
                if not isinstance(value, list):
                    return SetParametersResult(successful=False, reason="distortion_coeffs must be a list.")
                self.dist_coeffs = np.array(value)
                self.get_logger().info("Updated distortion_coeffs")

            elif name in ['square_length', 'marker_length']:
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name, float(value))
                self.get_logger().info(f"Updated {name}: {value}")
                self.create_charuco_board()

            elif name in ['board_squares_x', 'board_squares_y']:
                if not isinstance(value, int) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")
                self.create_charuco_board()

            elif name == 'min_corners_detected':
                if not isinstance(value, int) or value <= 0:
                    return SetParametersResult(successful=False, reason="min_corners_detected must be > 0.")
                self.min_corners_detected = value
                self.get_logger().info(f"Updated min_corners_detected: {value}")

            elif name == 'aruco_dict':
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason="aruco_dict must be a string.")
                self.aruco_dict_name = value
                self.get_logger().info(f"Updated aruco_dict: {value}")
                self.create_charuco_board()

            elif name == 'center_board_origin':
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason="center_board_origin must be a bool.")
                self.center_board_origin = value
                self.get_logger().info(f"Updated center_board_origin: {value}")

            elif name in ['publish_debug_image', 'publish_tf']:
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason=f"{name} must be a bool.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")

            elif name in ['world_frame', 'board_frame', 'drone_frame']:
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason=f"{name} must be a string.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")

        return SetParametersResult(successful=True)

    def destroy_node(self):
        """Clean up before shutting down."""
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CharuCoPoseEstimator()
    except Exception as e:
        print(f"[FATAL] CharuCoPoseEstimator failed to initialize: {e}", file=sys.stderr)
        rclpy.shutdown()
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted with Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()