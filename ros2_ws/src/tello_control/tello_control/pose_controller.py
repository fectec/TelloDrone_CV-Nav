#!/usr/bin/env python3

import sys
import math
from simple_pid import PID

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PoseStamped, Twist, Pose
from std_msgs.msg import Empty
from std_srvs.srv import Trigger

class PoseController(Node):
    """
    Controls drone movement based on pose feedback and requests waypoints from service.
    """
    def __init__(self):
        super().__init__('pose_controller')
        
        # Declare parameters
        self.declare_parameter('update_rate', 30.0)                    # Hz
        self.declare_parameter('pose_topic', 'drone/pose')             
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('waypoint_service', 'get_next_waypoint')
        
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
        self.declare_parameter('yaw_tolerance', 0.1)                   # rad
        
        # Control enable/disable
        self.declare_parameter('control_enabled', True)                
        
        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.waypoint_service_name = self.get_parameter('waypoint_service').value
        
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
        
        self.control_enabled = self.get_parameter('control_enabled').value
        
        # Timer for periodic control
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on-set-parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Validate initial parameters
        init_params = [
            Parameter('update_rate', Parameter.Type.DOUBLE, self.update_rate),
            Parameter('pose_topic', Parameter.Type.STRING, self.pose_topic),
            Parameter('cmd_vel_topic', Parameter.Type.STRING, self.cmd_vel_topic),
            Parameter('waypoint_service', Parameter.Type.STRING, self.waypoint_service_name),
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
            Parameter('control_enabled', Parameter.Type.BOOL, self.control_enabled),
        ]
        
        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize PID controllers
        self.create_pid_controllers()
        
        # Create service client
        self.waypoint_client = self.create_client(Trigger, self.waypoint_service_name)
        
        # State variables
        self.current_pose = None
        self.current_waypoint = None
        self.waypoint_requested = False
        self.mission_complete = False
        self.landing_triggered = False
        
        # Create subscriptions
        self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            qos_profile=qos.QoSProfile(
                depth=10,
                reliability=qos.ReliabilityPolicy.RELIABLE,
                durability=qos.DurabilityPolicy.VOLATILE
            )
        )
        
        # Create publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            qos_profile=qos.QoSProfile(
                depth=10, 
                reliability=qos.ReliabilityPolicy.RELIABLE,
                durability=qos.DurabilityPolicy.VOLATILE
            )
        )
        
        # Create publisher for TelloDriver landing command
        self.land_pub = self.create_publisher(
            Empty, 
            'tello/land', 
            qos_profile=qos.QoSProfile(
                depth=5,
                reliability=qos.ReliabilityPolicy.RELIABLE,
                durability=qos.DurabilityPolicy.VOLATILE
            )
        )
        
        self.get_logger().info("PoseController Start.")
        
        # Wait for service and request first waypoint
        self.request_first_waypoint()
    
    def request_first_waypoint(self):
        """Wait for service and request the first waypoint."""
        def request_callback():
            if not self.waypoint_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for waypoint service...")
                self.create_timer(1.0, request_callback, single_use=True)
                return
            
            self.request_next_waypoint()
        
        request_callback()
    
    def request_next_waypoint(self):
        """Request the next waypoint from the service."""
        if self.mission_complete or self.waypoint_requested:
            return
        
        self.waypoint_requested = True
        request = Trigger.Request()
        future = self.waypoint_client.call_async(request)
        future.add_done_callback(self.waypoint_response_callback)
    
    def waypoint_response_callback(self, future):
        """Handle waypoint service response."""
        self.waypoint_requested = False
        
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return
        
        if not response.success:
            self.get_logger().info(f"Waypoint service: {response.message}")
            if "Mission complete" in response.message:
                self.mission_complete = True
                self.land_drone()
            return
        
        # Parse waypoint from response message
        try:
            parts = response.message.split(',')
            if len(parts) != 7:
                raise ValueError("Invalid waypoint format.")
            
            waypoint = Pose()
            waypoint.position.x = float(parts[0])
            waypoint.position.y = float(parts[1])
            waypoint.position.z = float(parts[2])
            waypoint.orientation.x = float(parts[3])
            waypoint.orientation.y = float(parts[4])
            waypoint.orientation.z = float(parts[5])
            waypoint.orientation.w = float(parts[6])
            
            self.current_waypoint = waypoint
            self.get_logger().info(
                f"Received waypoint: ({waypoint.position.x:.2f}, {waypoint.position.y:.2f}, {waypoint.position.z:.2f})"
            )
            
        except (ValueError, IndexError) as e:
            self.get_logger().error(f"Failed to parse waypoint: {e}")
    
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
    
    def timer_callback(self):
        """Main control loop."""
        if self.mission_complete and not self.landing_triggered:
            return
            
        if self.current_pose is None or self.current_waypoint is None:
            return
        
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
        target_x = self.current_waypoint.position.x
        target_y = self.current_waypoint.position.y
        target_z = self.current_waypoint.position.z
        target_yaw = self.quaternion_to_yaw(
            self.current_waypoint.orientation.x,
            self.current_waypoint.orientation.y,
            self.current_waypoint.orientation.z,
            self.current_waypoint.orientation.w
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
                
        # Check if waypoint reached
        pos_error = math.sqrt(x_err**2 + y_err**2 + z_err**2)
        yaw_error = abs(yaw_err)
        
        waypoint_reached = (pos_error < self.position_tolerance and 
                           yaw_error < self.yaw_tolerance)
        
        if waypoint_reached:
            self.get_logger().info(f"Waypoint reached! Requesting next waypoint...")
            self.current_waypoint = None  # Clear current waypoint
            self.request_next_waypoint()  # Request next waypoint
            return
        
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
            f"Tracking waypoint: pos_err={pos_error:.3f}m, yaw_err={yaw_error:.3f}rad",
            throttle_duration_sec=2.0
        )

    def land_drone(self):
        """Land the drone using TelloDriver's landing functionality."""
        if self.landing_triggered:
            return
            
        self.landing_triggered = True
        self.get_logger().info("Mission complete! Triggering TelloDriver landing sequence...")
        
        # Stop sending velocity commands
        self.control_enabled = False
        
        # Send zero velocity command to stop movement
        vel_msg = Twist()
        self.cmd_vel_pub.publish(vel_msg)
        
        # Trigger TelloDriver's landing sequence
        land_msg = Empty()
        self.land_pub.publish(land_msg)
        
        self.get_logger().info("Landing command sent to TelloDriver.")
    
    def quaternion_to_yaw(self, x, y, z, w):
        """Extract yaw from quaternion."""
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
            
            elif name in ['pose_topic', 'cmd_vel_topic', 'waypoint_service']:
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason=f"{name} must be a string.")
                setattr(self, name.replace('waypoint_service', 'waypoint_service_name'), value)
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
            
            elif name in ['position_tolerance', 'yaw_tolerance']:
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name, float(value))
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name == 'control_enabled':
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason=f"{name} must be a bool.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")
        
        return SetParametersResult(successful=True)
    
    def destroy_node(self):
        """Clean up before shutting down."""
        # Send zero velocity before shutdown
        if self.control_enabled:
            vel_msg = Twist()
            self.cmd_vel_pub.publish(vel_msg)
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