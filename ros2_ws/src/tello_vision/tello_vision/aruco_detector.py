#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge, CvBridgeError

import sys
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import Int32MultiArray, Float32MultiArray, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker

from tf_transformations import quaternion_from_euler

class ArucoDetector(Node):
    """
    Detects ArUco markers from camera images (real or simulated).
    
    Subscribes to:
      - image topic (sensor_msgs/Image): Camera feed for marker detection
      - simulationTime (std_msgs/Float32): Simulation time when in simulation mode
    
    Publishes to:
      - aruco/poses (geometry_msgs/PoseArray):                  Array of marker poses in camera frame
      - aruco/ids (std_msgs/Int32MultiArray):                   Array of detected marker IDs
      - aruco/corners (std_msgs/Float32MultiArray):             Flattened array of corner coordinates
      - aruco/visualization (visualization_msgs/MarkerArray):   Visualization markers for RViz
      - aruco/debug_image (sensor_msgs/Image):                  Annotated image with detected markers
    """

    def __init__(self):
        super().__init__('aruco_detector_node')
        
        # Declare parameters
        self.declare_parameter('update_rate', 30.0)                                 # Hz
        self.declare_parameter('simulation', False)  
        self.declare_parameter('sim_image_topic', '/downward/image_raw') 
        self.declare_parameter('real_image_topic', '/downward/image_raw') 
        self.declare_parameter('dictionary_id', 2)  
        self.declare_parameter('marker_size', 0.17)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('visualize_axes', True)
        
        # Camera calibration parameters
        self.declare_parameter('camera_matrix', [500.0, 0.0, 320.0, 
                                                0.0, 500.0, 240.0, 
                                                0.0, 0.0, 1.0])
        self.declare_parameter('distortion_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
        
        # ArUco detection parameters
        self.declare_parameter('corner_refinement', 1) 
        self.declare_parameter('adaptive_thresh_constant', 7.0)
        self.declare_parameter('min_marker_perimeter_rate', 0.03)
        self.declare_parameter('max_marker_perimeter_rate', 4.0)
        
        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.simulation = self.get_parameter('simulation').value
        self.sim_image_topic = self.get_parameter('sim_image_topic').value
        self.real_image_topic = self.get_parameter('real_image_topic').value
        self.dictionary_id = self.get_parameter('dictionary_id').value
        self.marker_size = self.get_parameter('marker_size').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        self.visualize_axes = self.get_parameter('visualize_axes').value
        
        camera_matrix_flat = self.get_parameter('camera_matrix').value
        self.camera_matrix = np.array(camera_matrix_flat).reshape(3, 3)
        self.dist_coeffs = np.array(self.get_parameter('distortion_coeffs').value)
        
        self.corner_refinement = self.get_parameter('corner_refinement').value
        self.adaptive_thresh_constant = self.get_parameter('adaptive_thresh_constant').value
        self.min_marker_perimeter_rate = self.get_parameter('min_marker_perimeter_rate').value
        self.max_marker_perimeter_rate = self.get_parameter('max_marker_perimeter_rate').value
        
        # Timer for processing loop
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on‐set‐parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Immediately validate the initial values
        init_params = [
            Parameter('update_rate',                Parameter.Type.DOUBLE,          self.update_rate),
            Parameter('simulation',                 Parameter.Type.BOOL,            self.simulation),
            Parameter('sim_image_topic',            Parameter.Type.STRING,          self.sim_image_topic),
            Parameter('real_image_topic',           Parameter.Type.STRING,          self.real_image_topic),
            Parameter('dictionary_id',              Parameter.Type.INTEGER,         self.dictionary_id),
            Parameter('marker_size',                Parameter.Type.DOUBLE,          self.marker_size),
            Parameter('publish_debug_image',        Parameter.Type.BOOL,            self.publish_debug_image),
            Parameter('visualize_axes',             Parameter.Type.BOOL,            self.visualize_axes),
            Parameter('camera_matrix',              Parameter.Type.DOUBLE_ARRAY,    camera_matrix_flat),
            Parameter('distortion_coeffs',          Parameter.Type.DOUBLE_ARRAY,    self.dist_coeffs.tolist()),
            Parameter('corner_refinement',          Parameter.Type.INTEGER,         self.corner_refinement),
            Parameter('adaptive_thresh_constant',   Parameter.Type.DOUBLE,          self.adaptive_thresh_constant),
            Parameter('min_marker_perimeter_rate',  Parameter.Type.DOUBLE,          self.min_marker_perimeter_rate),
            Parameter('max_marker_perimeter_rate',  Parameter.Type.DOUBLE,          self.max_marker_perimeter_rate),
        ]
        
        result: SetParametersResult = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize variables
        self.image = None
        self.image_gray = None  
        self.image_header = None
        self.bridge = CvBridge()
        
        # Time tracking
        self.sim_time = None
        self.last_time = None
        
        # Initialize ArUco dictionary and detector
        self._init_aruco_dictionary()
        self._init_detector_parameters()
        
        # Limit logging frequency
        self.last_log_time = 0.0
        self.frame_count = 0
        
        # Publishers
        self.poses_pub = self.create_publisher(PoseArray,           'aruco/poses',          10)
        self.ids_pub = self.create_publisher(Int32MultiArray,       'aruco/ids',            10)
        self.corners_pub = self.create_publisher(Float32MultiArray, 'aruco/corners',        10)
        self.markers_pub = self.create_publisher(MarkerArray,       'aruco/visualization',  10)
        
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image,     'aruco/debug_image',    10)
        
        # Select image topic based on simulation parameter
        image_topic = self.sim_image_topic if self.simulation else self.real_image_topic
        
        # Subscriber for camera images
        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos.qos_profile_sensor_data
        )
        
        # Simulation time subscriber (only if in simulation mode)
        if self.simulation:
            self.create_subscription(
                Float32,
                'simulationTime',
                self.sim_time_callback,
                qos.qos_profile_sensor_data
            )
        
        self.get_logger().info("ArucoDetector Start.")
    
    def sim_time_callback(self, msg: Float32) -> None:
        """Update simulation time from CoppeliaSim."""
        self.sim_time = msg.data
    
    def image_callback(self, msg: Image) -> None:
        """Callback to convert ROS image to OpenCV format and store it."""
        try:
            if msg.encoding == 'mono8':
                self.image_gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                # Only convert to BGR if debug image is needed
                if self.publish_debug_image:
                    self.image = cv2.cvtColor(self.image_gray, cv2.COLOR_GRAY2BGR)
            elif msg.encoding == 'rgb8':
                # Convert RGB to grayscale for processing
                rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                self.image_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                if self.publish_debug_image:
                    self.image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                # Convert BGR to grayscale for processing
                bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.image_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                if self.publish_debug_image:
                    self.image = bgr_image
            else:
                # Fallback: try to get grayscale
                temp_image = self.bridge.imgmsg_to_cv2(msg)
                if len(temp_image.shape) == 2:
                    self.image_gray = temp_image
                    if self.publish_debug_image:
                        self.image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
                else:
                    self.image_gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
                    if self.publish_debug_image:
                        self.image = temp_image
            
            self.image_header = msg.header
            self.frame_count += 1
            
            # Log first frame info
            if self.frame_count == 1:
                self.get_logger().info(f"First image received! Size: {msg.width}x{msg.height}, Encoding: {msg.encoding}.")
                
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}. Image encoding: {msg.encoding}.")
            return
    
    def timer_callback(self) -> None:
        """Main timer function to process images and detect ArUco markers."""
        # Check if image has been received
        if self.image_gray is None:
            return
        
        # In simulation mode, check if time has progressed
        if self.simulation:
            if self.sim_time is None:
                return
            
            if self.last_time is None:
                self.last_time = self.sim_time
                return
            
            # Check if enough simulation time has passed
            dt = self.sim_time - self.last_time
            if dt < 1.0 / self.update_rate:
                return
            self.last_time = self.sim_time
        
        # Process the grayscale image
        processed_gray = self._preprocess_image(self.image_gray)
        
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            processed_gray, self.aruco_dict, parameters=self.detector_params
        )
        
        # Process detections
        if ids is not None and len(ids) > 0:
            self._process_detections(corners, ids)
            
            # Log detection info
            current_time = time.time()
            if current_time - self.last_log_time > 1.0:
                self.get_logger().info(
                    f"Detected {len(ids)} markers: {ids.flatten().tolist()}"
                )
                self.last_log_time = current_time
        else:
            # Publish empty arrays when no markers detected
            self._publish_empty_results()
            
            # Log no detection periodically
            current_time = time.time()
            if current_time - self.last_log_time > 2.0:
                self.get_logger().debug("No markers detected")
                self.last_log_time = current_time
        
        # Publish debug image if enabled
        if self.publish_debug_image and self.image is not None:
            self._publish_debug_image(corners, ids)
    
    def _preprocess_image(self, gray_image):
        """Preprocess grayscale image for better ArUco detection."""
        processed = gray_image.copy()
        
        if self.simulation:
            # Apply CLAHE to grayscale
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            processed = clahe.apply(processed)
            
            # Slight sharpening to improve marker edges
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        return processed
    
    def _process_detections(self, corners, ids):
        """Process detected markers and publish results."""
        # Prepare messages
        pose_array = PoseArray()
        pose_array.header = self.image_header
        pose_array.header.frame_id = self.image_header.frame_id if self.image_header.frame_id else "camera_frame"
        
        id_array = Int32MultiArray()
        id_array.data = ids.flatten().tolist()
        
        corners_array = Float32MultiArray()
        corners_flat = []
        
        marker_array = MarkerArray()
        
        # Process each detected marker
        for i, (corner, marker_id) in enumerate(zip(corners, ids)):
            marker_id = int(marker_id[0])
            
            # Flatten corner coordinates
            for point in corner[0]:
                corners_flat.extend([float(point[0]), float(point[1])])
            
            # Estimate pose if marker size is provided
            if self.marker_size > 0:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corner], self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                
                # Convert to ROS pose
                pose = Pose()
                pose.position.x = tvec[0]
                pose.position.y = tvec[1]
                pose.position.z = tvec[2]
                
                # Convert rotation vector to quaternion
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                # Extract Euler angles from rotation matrix
                sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
                singular = sy < 1e-6
                if not singular:
                    x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                    y = np.arctan2(-rotation_matrix[2,0], sy)
                    z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
                else:
                    x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                    y = np.arctan2(-rotation_matrix[2,0], sy)
                    z = 0
                
                q = quaternion_from_euler(x, y, z)
                pose.orientation.x = q[0]
                pose.orientation.y = q[1]
                pose.orientation.z = q[2]
                pose.orientation.w = q[3]
                
                pose_array.poses.append(pose)
                
                # Create visualization marker
                viz_marker = self._create_visualization_marker(
                    marker_id, pose, self.image_header, i
                )
                marker_array.markers.append(viz_marker)
        
        corners_array.data = corners_flat
        
        # Publish results
        self.poses_pub.publish(pose_array)
        self.ids_pub.publish(id_array)
        self.corners_pub.publish(corners_array)
        self.markers_pub.publish(marker_array)
    
    def _publish_empty_results(self):
        """Publish empty results when no markers are detected."""
        # Empty pose array
        pose_array = PoseArray()
        if self.image_header:
            pose_array.header = self.image_header
            pose_array.header.frame_id = self.image_header.frame_id if self.image_header.frame_id else "camera_frame"
        self.poses_pub.publish(pose_array)
        
        # Empty ID array
        id_array = Int32MultiArray()
        self.ids_pub.publish(id_array)
        
        # Empty corners array
        corners_array = Float32MultiArray()
        self.corners_pub.publish(corners_array)
        
        # Empty marker array
        marker_array = MarkerArray()
        self.markers_pub.publish(marker_array)
    
    def _create_visualization_marker(self, marker_id, pose, header, index):
        """Create a visualization marker for RViz."""
        marker = Marker()
        marker.header = header
        marker.header.frame_id = header.frame_id if header.frame_id else "camera_frame"
        marker.ns = "aruco_markers"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = self.marker_size
        marker.scale.y = self.marker_size
        marker.scale.z = 0.001  
        marker.color.a = 0.8
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        
        return marker
    
    def _publish_debug_image(self, corners, ids):
        """Publish annotated debug image."""
        debug_image = self.image.copy()
        
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
            
            # Draw axes if requested and camera is calibrated
            if self.visualize_axes and self.marker_size > 0:
                for corner in corners:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corner], self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    cv2.drawFrameAxes(
                        debug_image, self.camera_matrix, self.dist_coeffs,
                        rvecs[0], tvecs[0], self.marker_size * 0.5
                    )
            
            # Add text overlay for each marker
            for i, (corner, marker_id) in enumerate(zip(corners, ids)):
                corner_int = corner[0].astype(int)
                center = np.mean(corner_int, axis=0).astype(int)
                cv2.putText(
                    debug_image, f"ID:{marker_id[0]}", 
                    (center[0]-20, center[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
        
        # Add frame info
        cv2.putText(
            debug_image, f"Frame: {self.frame_count}, Mode: {'Sim' if self.simulation else 'Real'}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )
        
        # Convert and publish
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        if self.image_header:
            debug_msg.header = self.image_header
        self.debug_image_pub.publish(debug_msg)
    
    def _init_aruco_dictionary(self):
        """Initialize ArUco dictionary based on dictionary_id parameter."""
        dict_mapping = {
            0: cv2.aruco.DICT_4X4_50,
            1: cv2.aruco.DICT_4X4_100,
            2: cv2.aruco.DICT_4X4_250,
            3: cv2.aruco.DICT_4X4_1000,
            4: cv2.aruco.DICT_5X5_50,
            5: cv2.aruco.DICT_5X5_100,
            6: cv2.aruco.DICT_5X5_250,
            7: cv2.aruco.DICT_5X5_1000,
            8: cv2.aruco.DICT_6X6_50,
            9: cv2.aruco.DICT_6X6_100,
            10: cv2.aruco.DICT_6X6_250,
            11: cv2.aruco.DICT_6X6_1000,
            12: cv2.aruco.DICT_7X7_50,
            13: cv2.aruco.DICT_7X7_100,
            14: cv2.aruco.DICT_7X7_250,
            15: cv2.aruco.DICT_7X7_1000,
            16: cv2.aruco.DICT_ARUCO_ORIGINAL,
            17: cv2.aruco.DICT_APRILTAG_16h5,
            18: cv2.aruco.DICT_APRILTAG_25h9,
            19: cv2.aruco.DICT_APRILTAG_36h10,
            20: cv2.aruco.DICT_APRILTAG_36h11
        }
        
        if self.dictionary_id in dict_mapping:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_mapping[self.dictionary_id])
        else:
            self.get_logger().warn(f"Invalid dictionary_id: {self.dictionary_id}. Using DICT_5X5_50.")
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    
    def _get_dictionary_name(self, dict_id):
        """Get human-readable dictionary name."""
        names = {
            0: "DICT_4X4_50", 1: "DICT_4X4_100", 2: "DICT_4X4_250", 3: "DICT_4X4_1000",
            4: "DICT_5X5_50", 5: "DICT_5X5_100", 6: "DICT_5X5_250", 7: "DICT_5X5_1000",
            8: "DICT_6X6_50", 9: "DICT_6X6_100", 10: "DICT_6X6_250", 11: "DICT_6X6_1000",
            12: "DICT_7X7_50", 13: "DICT_7X7_100", 14: "DICT_7X7_250", 15: "DICT_7X7_1000",
            16: "DICT_ARUCO_ORIGINAL", 17: "DICT_APRILTAG_16h5", 18: "DICT_APRILTAG_25h9",
            19: "DICT_APRILTAG_36h10", 20: "DICT_APRILTAG_36h11"
        }
        return names.get(dict_id, "UNKNOWN")
    
    def _init_detector_parameters(self):
        """Initialize ArUco detector parameters."""
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # Basic parameters
        self.detector_params.adaptiveThreshConstant = self.adaptive_thresh_constant
        self.detector_params.minMarkerPerimeterRate = self.min_marker_perimeter_rate
        self.detector_params.maxMarkerPerimeterRate = self.max_marker_perimeter_rate
        
        # Corner refinement
        if self.corner_refinement == 0:
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        elif self.corner_refinement == 1:
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        elif self.corner_refinement == 2:
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        else:
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        # Specific adjustments for simulation
        if self.simulation:
            # More aggressive parameters for simulation/grayscale images
            self.detector_params.adaptiveThreshWinSizeMin = 3
            self.detector_params.adaptiveThreshWinSizeMax = 23
            self.detector_params.adaptiveThreshWinSizeStep = 10
            self.detector_params.perspectiveRemovePixelPerCell = 4
            self.detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
            self.detector_params.minCornerDistanceRate = 0.05
            self.detector_params.minDistanceToBorder = 3
            self.detector_params.minOtsuStdDev = 5.0
            self.detector_params.errorCorrectionRate = 0.6
    
    def parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        """Validates and applies updated node parameters."""
        for param in params:
            name = param.name
            value = param.value

            if name == 'update_rate':
                if not isinstance(value, (int, float)) or value <= 0.0:
                    return SetParametersResult(successful=False, reason="update_rate must be > 0.")
                self.update_rate = float(value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
                self.get_logger().info(f"update_rate updated: {self.update_rate} Hz.")

            elif name == 'simulation':
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason="simulation must be true or false.")
                self.simulation = value
                self._init_detector_parameters()
                self.get_logger().info(f"simulation mode updated: {self.simulation}.")

            elif name == 'sim_image_topic':
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason="sim_image_topic must be a string.")
                self.sim_image_topic = value
                self.get_logger().info(f"sim_image_topic updated: {self.sim_image_topic}")

            elif name == 'real_image_topic':
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason="real_image_topic must be a string.")
                self.real_image_topic = value
                self.get_logger().info(f"real_image_topic updated: {self.real_image_topic}")

            elif name == 'dictionary_id':
                if not isinstance(value, int) or value < 0 or value > 20:
                    return SetParametersResult(successful=False, reason="dictionary_id must be an integer between 0 and 20.")
                self.dictionary_id = value
                self._init_aruco_dictionary()
                self.get_logger().info(f"dictionary_id updated: {self._get_dictionary_name(self.dictionary_id)}.")

            elif name == 'marker_size':
                if not isinstance(value, (int, float)) or value < 0.0:
                    return SetParametersResult(successful=False, reason="marker_size must be >= 0.")
                self.marker_size = float(value)
                self.get_logger().info(f"marker_size updated: {self.marker_size} m.")

            elif name == 'corner_refinement':
                if not isinstance(value, int) or value not in [0, 1, 2]:
                    return SetParametersResult(successful=False, reason="corner_refinement must be 0, 1, or 2.")
                self.corner_refinement = value
                self._init_detector_parameters()
                self.get_logger().info(f"corner_refinement updated: {self.corner_refinement}")

            elif name == 'publish_debug_image':
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason="publish_debug_image must be true or false.")
                self.publish_debug_image = value
                if self.publish_debug_image and not hasattr(self, 'debug_image_pub'):
                    self.debug_image_pub = self.create_publisher(Image, 'aruco/debug_image', 10)
                self.get_logger().info(f"publish_debug_image updated: {self.publish_debug_image}")

            elif name == 'visualize_axes':
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason="visualize_axes must be true or false.")
                self.visualize_axes = value
                self.get_logger().info(f"visualize_axes updated: {self.visualize_axes}")

            elif name == 'adaptive_thresh_constant':
                if not isinstance(value, (int, float)):
                    return SetParametersResult(successful=False, reason="adaptive_thresh_constant must be a number.")
                self.adaptive_thresh_constant = float(value)
                self.get_logger().info(f"adaptive_thresh_constant updated: {self.adaptive_thresh_constant}")

            elif name == 'min_marker_perimeter_rate':
                if not isinstance(value, (int, float)) or value < 0.0:
                    return SetParametersResult(successful=False, reason="min_marker_perimeter_rate must be >= 0.")
                self.min_marker_perimeter_rate = float(value)
                self.get_logger().info(f"min_marker_perimeter_rate updated: {self.min_marker_perimeter_rate}")

            elif name == 'max_marker_perimeter_rate':
                if not isinstance(value, (int, float)) or value < 0.0:
                    return SetParametersResult(successful=False, reason="max_marker_perimeter_rate must be >= 0.")
                self.max_marker_perimeter_rate = float(value)
                self.get_logger().info(f"max_marker_perimeter_rate updated: {self.max_marker_perimeter_rate}")

        return SetParametersResult(successful=True)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ArucoDetector()
    except Exception as e:
        print(f"[FATAL] ArucoDetector failed to initialize: {e}.", file=sys.stderr)
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