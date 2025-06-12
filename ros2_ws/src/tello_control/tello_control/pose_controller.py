#!/usr/bin/env python3

import sys
import math
import numpy as np
from simple_pid import PID

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PoseStamped, Twist, PoseArray
from std_msgs.msg import Bool, Int32

from tello_utils.matrix_helpers import matrix_to_vecs

class PoseController(Node):
    """
    Controls drone movement based on pose feedback and waypoint commands.
    Subscribes to pose estimates and waypoint arrays, publishes velocity commands.
    """
    def __init__(self):
        super().__init__('pose_controller')
        
        # Declare parameters
        self.declare_parameter('update_rate', 30.0)                    # Hz
        self.declare_parameter('pose_topic', 'drone/pose')             
        self.declare_parameter('waypoints_topic', 'waypoints')         
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')             
        
        # PID gain parameters
        self.declare_parameter('pid_x_kp', 1.0)
        self.declare_parameter('pid_x_ki', 0.0)
        self.declare_parameter('pid_x_kd', 0.0)
        self.declare_parameter('pid_y_kp', 1.0)
        self.declare_parameter('pid_y_ki', 0.0)
        self.declare_parameter('pid_y_kd', 0.0)
        self.declare_parameter('pid_z_kp', 1.0)
        self.declare_parameter('pid_z_ki', 0.0)
        self.declare_parameter('pid_z_kd', 0.0)
        self.declare_parameter('pid_yaw_kp', 1.0)
        self.declare_parameter('pid_yaw_ki', 0.0)
        self.declare_parameter('pid_yaw_kd', 0.0)
        
        # Control limits
        self.declare_parameter('max_linear_vel', 1.0)                  # m/s
        self.declare_parameter('max_angular_vel', 1.0)                 # rad/s
        
        # Waypoint following parameters
        self.declare_parameter('position_tolerance', 0.1)              # m
        self.declare_parameter('yaw_tolerance', 1.0)                   # rad
        self.declare_parameter('stability_threshold', 0.05)           
        self.declare_parameter('auto_advance', True)                  
        self.declare_parameter('loop_waypoints', False)                
        
        # Control enable/disable
        self.declare_parameter('control_enabled', True)                # Enable/disable control output
        
        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.waypoints_topic = self.get_parameter('waypoints_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
        self.stability_threshold = self.get_parameter('stability_threshold').value
        self.auto_advance = self.get_parameter('auto_advance').value
        self.loop_waypoints = self.get_parameter('loop_waypoints').value
        
        self.control_enabled = self.get_parameter('control_enabled').value
        
        # Timer for periodic control
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on-set-parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Validate initial parameters
        init_params = [
            Parameter('update_rate', Parameter.Type.DOUBLE, self.update_rate),
            Parameter('pose_topic', Parameter.Type.STRING, self.pose_topic),
            Parameter('waypoints_topic', Parameter.Type.STRING, self.waypoints_topic),
            Parameter('cmd_vel_topic', Parameter.Type.STRING, self.cmd_vel_topic),
            Parameter('pid_x_kp', Parameter.Type.DOUBLE, self.get_parameter('pid_x_kp').value),
            Parameter('pid_x_ki', Parameter.Type.DOUBLE, self.get_parameter('pid_x_ki').value),
            Parameter('pid_x_kd', Parameter.Type.DOUBLE, self.get_parameter('pid_x_kd').value),
            Parameter('pid_y_kp', Parameter.Type.DOUBLE, self.get_parameter('pid_y_kp').value),
            Parameter('pid_y_ki', Parameter.Type.DOUBLE, self.get_parameter('pid_y_ki').value),
            Parameter('pid_y_kd', Parameter.Type.DOUBLE, self.get_parameter('pid_y_kd').value),
            Parameter('pid_z_kp', Parameter.Type.DOUBLE, self.get_parameter('pid_z_kp').value),
            Parameter('pid_z_ki', Parameter.Type.DOUBLE, self.get_parameter('pid_z_ki').value),
            Parameter('pid_z_kd', Parameter.Type.DOUBLE, self.get_parameter('pid_z_kd').value),
            Parameter('pid_yaw_kp', Parameter.Type.DOUBLE, self.get_parameter('pid_yaw_kp').value),
            Parameter('pid_yaw_ki', Parameter.Type.DOUBLE, self.get_parameter('pid_yaw_ki').value),
            Parameter('pid_yaw_kd', Parameter.Type.DOUBLE, self.get_parameter('pid_yaw_kd').value),
            Parameter('max_linear_vel', Parameter.Type.DOUBLE, self.max_linear_vel),
            Parameter('max_angular_vel', Parameter.Type.DOUBLE, self.max_angular_vel),
            Parameter('position_tolerance', Parameter.Type.DOUBLE, self.position_tolerance),
            Parameter('yaw_tolerance', Parameter.Type.DOUBLE, self.yaw_tolerance),
            Parameter('stability_threshold', Parameter.Type.DOUBLE, self.stability_threshold),
            Parameter('auto_advance', Parameter.Type.BOOL, self.auto_advance),
            Parameter('loop_waypoints', Parameter.Type.BOOL, self.loop_waypoints),
            Parameter('control_enabled', Parameter.Type.BOOL, self.control_enabled),
        ]
        
        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize PID controllers
        self.create_pid_controllers()
        
        # State variables
        self.current_pose = None
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.last_control_outputs = (0.0, 0.0, 0.0, 0.0)
        
        # Create subscriptions
        self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10
        )
        
        self.create_subscription(
            PoseArray,
            self.waypoints_topic,
            self.waypoints_callback,
            10
        )
        
        # Create publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10
        )
        
        self.waypoint_reached_pub = self.create_publisher(
            Bool,
            'waypoint_reached',
            10
        )
        
        self.current_waypoint_pub = self.create_publisher(
            Int32,
            'current_waypoint_index',
            10
        )
        
        self.get_logger().info("PoseController Start.")
    
    def create_pid_controllers(self):
        """Create or recreate PID controllers with current parameters."""
        self.x_pid = PID(
            self.get_parameter('pid_x_kp').value,
            self.get_parameter('pid_x_ki').value,
            self.get_parameter('pid_x_kd').value,
            setpoint=0,
            output_limits=(-self.max_linear_vel, self.max_linear_vel)
        )
        
        self.y_pid = PID(
            self.get_parameter('pid_y_kp').value,
            self.get_parameter('pid_y_ki').value,
            self.get_parameter('pid_y_kd').value,
            setpoint=0,
            output_limits=(-self.max_linear_vel, self.max_linear_vel)
        )
        
        self.z_pid = PID(
            self.get_parameter('pid_z_kp').value,
            self.get_parameter('pid_z_ki').value,
            self.get_parameter('pid_z_kd').value,
            setpoint=0,
            output_limits=(-self.max_linear_vel, self.max_linear_vel)
        )
        
        self.yaw_pid = PID(
            self.get_parameter('pid_yaw_kp').value,
            self.get_parameter('pid_yaw_ki').value,
            self.get_parameter('pid_yaw_kd').value,
            setpoint=0,
            output_limits=(-self.max_angular_vel, self.max_angular_vel)
        )
    
    def pose_callback(self, msg):
        """Store the latest pose estimate."""
        self.current_pose = msg
    
    def waypoints_callback(self, msg):
        """Update waypoint list and reset index."""
        self.waypoints = msg.poses
        self.current_waypoint_idx = 0
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints.")
    
    def timer_callback(self):
        """Main control loop."""
        # Check prerequisites
        if self.current_pose is None:
            return
        
        if not self.waypoints:
            # Send zero velocity if no waypoints
            if self.control_enabled:
                self.publish_zero_velocity()
            return
        
        # Get current waypoint
        if self.current_waypoint_idx >= len(self.waypoints):
            if self.loop_waypoints and len(self.waypoints) > 0:
                self.current_waypoint_idx = 0
            else:
                # Mission complete, send zero velocity
                if self.control_enabled:
                    self.publish_zero_velocity()
                return
        
        target_pose = self.waypoints[self.current_waypoint_idx]
        
        # Extract current position and yaw
        current_x = self.current_pose.pose.position.x
        current_y = self.current_pose.pose.position.y
        current_z = self.current_pose.pose.position.z
        current_yaw = self.quaternion_to_yaw(
            self.current_pose.pose.orientation.x,
            self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w
        )
        
        # Extract target position and yaw
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        target_z = target_pose.position.z
        target_yaw = self.quaternion_to_yaw(
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w
        )
        
        # Compute errors
        x_err = target_x - current_x
        y_err = target_y - current_y
        z_err = target_z - current_z
        yaw_err = self.normalize_angle(target_yaw - current_yaw)
        
        # Convert to local frame
        x_err_local, y_err_local = self.global_to_local(x_err, y_err, -current_yaw)
        
        # Compute control outputs
        x_control = self.x_pid(x_err_local)
        y_control = self.y_pid(-y_err_local)
        z_control = self.z_pid(-z_err)
        yaw_control = self.yaw_pid(yaw_err)
        
        self.last_control_outputs = (x_control, y_control, z_control, yaw_control)
        
        # Check if waypoint reached
        pos_error = math.sqrt(x_err**2 + y_err**2 + z_err**2)
        yaw_error = abs(yaw_err)
        controls_stable = all(abs(c) < self.stability_threshold for c in self.last_control_outputs)
        
        waypoint_reached = (pos_error < self.position_tolerance and 
                           yaw_error < self.yaw_tolerance and 
                           controls_stable)
        
        # Publish waypoint reached status
        reached_msg = Bool()
        reached_msg.data = waypoint_reached
        self.waypoint_reached_pub.publish(reached_msg)
        
        # Publish current waypoint index
        idx_msg = Int32()
        idx_msg.data = self.current_waypoint_idx
        self.current_waypoint_pub.publish(idx_msg)
        
        # Auto-advance if enabled and waypoint reached
        if waypoint_reached and self.auto_advance:
            self.current_waypoint_idx += 1
            self.get_logger().info(f"Waypoint {self.current_waypoint_idx} reached. Advancing.")
        
        # Publish velocity command
        if self.control_enabled:
            vel_msg = Twist()
            vel_msg.linear.x = x_control
            vel_msg.linear.y = y_control
            vel_msg.linear.z = z_control
            vel_msg.angular.z = yaw_control
            self.cmd_vel_pub.publish(vel_msg)
        
        # Log with throttle
        self.get_logger().info(
            f"Waypoint {self.current_waypoint_idx}: "
            f"pos_err={pos_error:.3f}m, yaw_err={yaw_error:.3f}rad",
            throttle_duration_sec=2.0
        )
    
    def publish_zero_velocity(self):
        """Publish zero velocity command."""
        vel_msg = Twist()
        self.cmd_vel_pub.publish(vel_msg)
    
    def quaternion_to_yaw(self, x, y, z, w):
        """Extract yaw from quaternion."""
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def global_to_local(self, x_global, y_global, yaw):
        """Convert global coordinates to local frame."""
        c = math.cos(yaw)
        s = math.sin(yaw)
        x_local = c * x_global + s * y_global
        y_local = -s * x_global + c * y_global
        return x_local, y_local
    
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
            
            elif name in ['pose_topic', 'waypoints_topic', 'cmd_vel_topic']:
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason=f"{name} must be a string.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name.startswith('pid_'):
                if not isinstance(value, (int, float)):
                    return SetParametersResult(successful=False, reason=f"{name} must be a number.")
                # Recreate PID controllers with new gains
                self.create_pid_controllers()
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name in ['max_linear_vel', 'max_angular_vel']:
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name, float(value))
                # Update PID output limits
                self.create_pid_controllers()
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name in ['position_tolerance', 'yaw_tolerance', 'stability_threshold']:
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name, float(value))
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name in ['auto_advance', 'loop_waypoints', 'control_enabled']:
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason=f"{name} must be a bool.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")
        
        return SetParametersResult(successful=True)
    
    def destroy_node(self):
        """Clean up before shutting down."""
        # Send zero velocity before shutdown
        if self.control_enabled:
            self.publish_zero_velocity()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PoseController()
    except Exception as e:
        print(f"[FATAL] PoseController failed to initialize: {e}", file=sys.stderr)
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