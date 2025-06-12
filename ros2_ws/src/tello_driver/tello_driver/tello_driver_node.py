#!/usr/bin/env python3

import sys
import time
import threading

import cv2
from cv_bridge import CvBridge
from djitellopy import Tello
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool, Float32


class TelloDriver(Node):
    """
    Unified driver for DJI Tello drone that handles both camera streaming and control.
    
    Manages a single Tello connection to:
    - Stream downward camera images
    - Convert cmd_vel to RC commands
    - Handle takeoff/land/emergency commands
    
    This prevents connection conflicts between separate camera and control nodes.
    """
    def __init__(self):
        super().__init__('tello_driver')
        
        # Declare parameters
        # Connection
        self.declare_parameter('auto_connect', True)
        self.declare_parameter('retry_count', 3)
        
        # Camera parameters
        self.declare_parameter('camera_rate', 30.0)                    # Hz
        self.declare_parameter('image_topic', '/downward/image_raw')
        self.declare_parameter('enable_camera', True)
        
        # Control parameters
        self.declare_parameter('control_rate', 50.0)                   # Hz
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('enable_control', True)
        
        # Velocity scaling factors
        self.declare_parameter('linear_scale_xy', 10.0)
        self.declare_parameter('linear_scale_z', 10.0)
        self.declare_parameter('angular_scale_z', 10.0)
        
        # Safety parameters
        self.declare_parameter('auto_takeoff_height', 0.5)            # m
        self.declare_parameter('cmd_timeout', 0.5)                    # s
        
        # Retrieve parameters
        self.auto_connect = self.get_parameter('auto_connect').value
        self.retry_count = self.get_parameter('retry_count').value
        
        self.camera_rate = self.get_parameter('camera_rate').value
        self.image_topic = self.get_parameter('image_topic').value
        self.enable_camera = self.get_parameter('enable_camera').value
        
        self.control_rate = self.get_parameter('control_rate').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.enable_control = self.get_parameter('enable_control').value
        
        self.linear_scale_xy = self.get_parameter('linear_scale_xy').value
        self.linear_scale_z = self.get_parameter('linear_scale_z').value
        self.angular_scale_z = self.get_parameter('angular_scale_z').value
        
        self.auto_takeoff_height = self.get_parameter('auto_takeoff_height').value
        self.cmd_timeout = self.get_parameter('cmd_timeout').value
        
        # Create timers
        if self.enable_camera:
            self.camera_timer = self.create_timer(1.0 / self.camera_rate, self.camera_callback)
        
        if self.enable_control:
            self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_callback)
        
        # Register parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Validate initial parameters
        init_params = [
            Parameter('auto_connect', Parameter.Type.BOOL, self.auto_connect),
            Parameter('retry_count', Parameter.Type.INTEGER, self.retry_count),
            Parameter('camera_rate', Parameter.Type.DOUBLE, self.camera_rate),
            Parameter('image_topic', Parameter.Type.STRING, self.image_topic),
            Parameter('enable_camera', Parameter.Type.BOOL, self.enable_camera),
            Parameter('control_rate', Parameter.Type.DOUBLE, self.control_rate),
            Parameter('cmd_vel_topic', Parameter.Type.STRING, self.cmd_vel_topic),
            Parameter('enable_control', Parameter.Type.BOOL, self.enable_control),
            Parameter('linear_scale_xy', Parameter.Type.DOUBLE, self.linear_scale_xy),
            Parameter('linear_scale_z', Parameter.Type.DOUBLE, self.linear_scale_z),
            Parameter('angular_scale_z', Parameter.Type.DOUBLE, self.angular_scale_z),
            Parameter('auto_takeoff_height', Parameter.Type.DOUBLE, self.auto_takeoff_height),
            Parameter('cmd_timeout', Parameter.Type.DOUBLE, self.cmd_timeout),
        ]
        
        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize state
        self.tello = None
        self.connected = False
        self.is_flying = False
        self._connection_lock = threading.Lock()
        
        # Camera state
        self.bridge = CvBridge()
        self.crop_start_row = None
        
        # Control state
        self.current_cmd = Twist()
        self.last_cmd_time = self.get_clock().now()
        
        # Create publishers
        if self.enable_camera:
            self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        
        self.battery_pub = self.create_publisher(Float32, 'tello/battery', 10)
        self.connected_pub = self.create_publisher(Bool, 'tello/connected', 10)
        self.flying_pub = self.create_publisher(Bool, 'tello/flying', 10)
        
        # Create subscriptions
        if self.enable_control:
            self.create_subscription(
                Twist,
                self.cmd_vel_topic,
                self.cmd_vel_callback,
                10
            )
            
            self.create_subscription(Empty, 'tello/takeoff', self.takeoff_callback, 10)
            self.create_subscription(Empty, 'tello/land', self.land_callback, 10)
            self.create_subscription(Empty, 'tello/emergency', self.emergency_callback, 10)
        
        # Auto-connect if enabled
        if self.auto_connect:
            self.connect_tello()
        
        self.get_logger().info("TelloDriver Start.")
    
    def connect_tello(self):
        """Connect to Tello drone and setup video stream."""
        with self._connection_lock:
            if self.connected:
                self.get_logger().warn("Already connected to Tello.")
                return
            
            success = False
            for attempt in range(1, self.retry_count + 1):
                try:
                    self.get_logger().info(f"Connecting to Tello... (attempt {attempt}/{self.retry_count})")
                    self.tello = Tello()
                    self.tello.connect()
                    
                    battery = self.tello.get_battery()
                    self.get_logger().info(f"Connected to Tello! Battery: {battery}%")
                    
                    # Setup camera if enabled
                    if self.enable_camera:
                        self.tello.set_video_direction(Tello.CAMERA_DOWNWARD)
                        self.tello.streamon()
                        self.get_logger().info("Downward camera stream started.")
                    
                    self.connected = True
                    success = True
                    break
                    
                except Exception as e:
                    self.get_logger().warn(f"Connection failed: {e}")
                    time.sleep(1)
            
            if not success:
                self.tello = None
                raise RuntimeError("Failed to connect to Tello after multiple attempts.")
    
    def disconnect_tello(self):
        """Disconnect from Tello drone."""
        with self._connection_lock:
            if not self.connected or self.tello is None:
                return
            
            try:
                # Land if flying
                if self.is_flying:
                    self.get_logger().info("Landing before disconnect...")
                    self.tello.land()
                    self.is_flying = False
                
                # Stop video stream
                if self.enable_camera:
                    self.tello.streamoff()
                
                # Disconnect
                self.tello.end()
                self.connected = False
                self.get_logger().info("Disconnected from Tello.")
                
            except Exception as e:
                self.get_logger().error(f"Error during disconnect: {e}")
            finally:
                self.tello = None
    
    def camera_callback(self):
        """Process and publish camera frames."""
        if not self.connected or self.tello is None:
            return
        
        try:
            frame = self.tello.get_frame_read().frame
            
            if frame is None or frame.size == 0:
                self.get_logger().warn("Empty or invalid frame.", throttle_duration_sec=5.0)
                return
            
            # Detect crop point only once (for green overlay removal)
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
            
            # Publish
            msg = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def control_callback(self):
        """Send RC commands and publish status."""
        # Publish status
        connected_msg = Bool()
        connected_msg.data = self.connected
        self.connected_pub.publish(connected_msg)
        
        flying_msg = Bool()
        flying_msg.data = self.is_flying
        self.flying_pub.publish(flying_msg)
        
        # Publish battery if connected
        if self.connected and self.tello is not None:
            try:
                battery = self.tello.get_battery()
                battery_msg = Float32()
                battery_msg.data = float(battery)
                self.battery_pub.publish(battery_msg)
            except:
                pass
        
        # Send RC commands if flying
        if not self.connected or self.tello is None or not self.is_flying:
            return
        
        # Check for command timeout
        time_since_cmd = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if time_since_cmd > self.cmd_timeout:
            # Send zero velocities if timeout
            self.send_rc_control(0, 0, 0, 0)
            return
        
        # Convert cmd_vel to RC commands
        left_right = int(self.current_cmd.linear.y * self.linear_scale_xy)
        forward_backward = int(self.current_cmd.linear.x * self.linear_scale_xy)
        up_down = int(self.current_cmd.linear.z * self.linear_scale_z)
        yaw = int(self.current_cmd.angular.z * self.angular_scale_z)
        
        # Clamp to valid range
        left_right = max(-10, min(10, left_right))
        forward_backward = max(-10, min(10, forward_backward))
        up_down = max(-10, min(10, up_down))
        yaw = max(-10, min(10, yaw))
        
        # Send RC control
        self.send_rc_control(left_right, forward_backward, up_down, yaw)
    
    def send_rc_control(self, left_right, forward_backward, up_down, yaw):
        """Send RC control command to Tello."""
        try:
            self.tello.send_rc_control(left_right, forward_backward, up_down, yaw)
            
            # Log with throttle
            self.get_logger().info(
                f"RC: lr={left_right}, fb={forward_backward}, ud={up_down}, yaw={yaw}",
                throttle_duration_sec=5.0
            )
            
        except Exception as e:
            self.get_logger().error(f"Failed to send RC control: {e}")
    
    def cmd_vel_callback(self, msg):
        """Store latest velocity command."""
        self.current_cmd = msg
        self.last_cmd_time = self.get_clock().now()
    
    def takeoff_callback(self, msg):
        """Handle takeoff command."""
        if not self.connected or self.tello is None:
            self.get_logger().error("Cannot takeoff - not connected to Tello.")
            return
        
        if self.is_flying:
            self.get_logger().warn("Already flying.")
            return
        
        try:
            self.get_logger().info("Taking off...")
            self.tello.takeoff()
            self.is_flying = True
            
            # Move up to desired height if needed
            if self.auto_takeoff_height > 0.3:  # Tello takeoff is ~0.3m
                height_diff = int((self.auto_takeoff_height - 0.3) * 100)  # cm
                if height_diff > 20:  # Only move if significant
                    self.tello.move_up(height_diff)
            
            self.get_logger().info("Takeoff complete.")
            
        except Exception as e:
            self.get_logger().error(f"Takeoff failed: {e}")
            self.is_flying = False
    
    def land_callback(self, msg):
        """Handle land command."""
        if not self.connected or self.tello is None:
            self.get_logger().error("Cannot land - not connected to Tello.")
            return
        
        if not self.is_flying:
            self.get_logger().warn("Not flying.")
            return
        
        try:
            self.get_logger().info("Landing...")
            self.tello.land()
            self.is_flying = False
            self.get_logger().info("Landed.")
            
        except Exception as e:
            self.get_logger().error(f"Landing failed: {e}")
    
    def emergency_callback(self, msg):
        """Handle emergency stop command."""
        if not self.connected or self.tello is None:
            self.get_logger().error("Cannot emergency stop - not connected to Tello.")
            return
        
        try:
            self.get_logger().warn("EMERGENCY STOP!")
            self.tello.emergency()
            self.is_flying = False
            
        except Exception as e:
            self.get_logger().error(f"Emergency stop failed: {e}")
    
    def parameter_callback(self, params):
        """Handle parameter updates."""
        for param in params:
            name = param.name
            value = param.value
            
            if name in ['auto_connect', 'enable_camera', 'enable_control']:
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason=f"{name} must be a bool.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name == 'retry_count':
                if not isinstance(value, int) or value < 0:
                    return SetParametersResult(successful=False, reason="retry_count must be >= 0.")
                self.retry_count = value
                self.get_logger().info(f"Updated retry_count: {value}")
            
            elif name in ['camera_rate', 'control_rate']:
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name, float(value))
                # Recreate timer
                if name == 'camera_rate' and self.enable_camera:
                    self.camera_timer.cancel()
                    self.camera_timer = self.create_timer(1.0 / self.camera_rate, self.camera_callback)
                elif name == 'control_rate' and self.enable_control:
                    self.control_timer.cancel()
                    self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_callback)
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name in ['image_topic', 'cmd_vel_topic']:
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason=f"{name} must be a string.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name in ['linear_scale_xy', 'linear_scale_z', 'angular_scale_z']:
                if not isinstance(value, (int, float)):
                    return SetParametersResult(successful=False, reason=f"{name} must be a number.")
                setattr(self, name, float(value))
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name in ['auto_takeoff_height', 'cmd_timeout']:
                if not isinstance(value, (int, float)) or value < 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be >= 0.")
                setattr(self, name, float(value))
                self.get_logger().info(f"Updated {name}: {value}")
        
        return SetParametersResult(successful=True)
    
    def destroy_node(self):
        """Clean up before shutting down."""
        # Send zero velocities if flying
        if self.connected and self.tello is not None and self.is_flying:
            try:
                self.send_rc_control(0, 0, 0, 0)
            except:
                pass
        
        # Disconnect from Tello
        self.disconnect_tello()
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TelloDriver()
    except Exception as e:
        print(f"[FATAL] TelloDriver failed to initialize: {e}", file=sys.stderr)
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