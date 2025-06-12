#!/usr/bin/env python3

import sys
import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header


class WaypointsPublisher(Node):
    """
    Publishes waypoint arrays for drone trajectory following.
    
    Supports multiple predefined trajectories and custom waypoint lists.
    Waypoints are published as PoseArray messages containing position and orientation.
    """
    def __init__(self):
        super().__init__('waypoints_publisher')
        
        # Declare parameters
        self.declare_parameter('update_rate', 1.0)                              # Hz
        self.declare_parameter('waypoints_topic', 'waypoints')        
        self.declare_parameter('frame_id', 'charuco_board')            
        
        # Trajectory selection
        self.declare_parameter('trajectory_type', 'custom')                     # hover, square, circle, figure8, custom
        self.declare_parameter('publish_once', False)                           # Publish once or continuously
        
        # Common trajectory parameters
        self.declare_parameter('height', 1.0)
        self.declare_parameter('size', 1.0)                            
        self.declare_parameter('num_points', 8)                        
        
        # Custom waypoints (x, y, z, yaw) - flat list
        self.declare_parameter('custom_waypoints', [0.45, -1.2, 0.5, 0.0])      # [x1,y1,z1,yaw1, x2,y2,z2,yaw2, ...]
        
        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.waypoints_topic = self.get_parameter('waypoints_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        
        self.trajectory_type = self.get_parameter('trajectory_type').value
        self.publish_once = self.get_parameter('publish_once').value
        
        self.height = self.get_parameter('height').value
        self.size = self.get_parameter('size').value
        self.num_points = self.get_parameter('num_points').value
        
        self.custom_waypoints = self.get_parameter('custom_waypoints').value
        
        # Timer for periodic publishing
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Validate initial parameters
        init_params = [
            Parameter('update_rate', Parameter.Type.DOUBLE, self.update_rate),
            Parameter('waypoints_topic', Parameter.Type.STRING, self.waypoints_topic),
            Parameter('frame_id', Parameter.Type.STRING, self.frame_id),
            Parameter('trajectory_type', Parameter.Type.STRING, self.trajectory_type),
            Parameter('publish_once', Parameter.Type.BOOL, self.publish_once),
            Parameter('height', Parameter.Type.DOUBLE, self.height),
            Parameter('size', Parameter.Type.DOUBLE, self.size),
            Parameter('num_points', Parameter.Type.INTEGER, self.num_points),
            Parameter('custom_waypoints', Parameter.Type.DOUBLE_ARRAY, self.custom_waypoints),
        ]
        
        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Create publisher
        self.waypoints_pub = self.create_publisher(
            PoseArray,
            self.waypoints_topic,
            10
        )
        
        # State
        self.published = False
        
        self.get_logger().info(f"WaypointsPublisher Start. Trajectory: {self.trajectory_type}")
    
    def timer_callback(self):
        """Publish waypoints based on selected trajectory."""
        if self.publish_once and self.published:
            return
        
        # Generate waypoints based on trajectory type
        waypoints = self.generate_waypoints()
        
        if not waypoints:
            self.get_logger().warn("No waypoints generated.")
            return
        
        # Create PoseArray message
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.poses = waypoints
        
        # Publish
        self.waypoints_pub.publish(msg)
        self.published = True
        
        self.get_logger().info(
            f"Published {len(waypoints)} waypoints for {self.trajectory_type} trajectory.",
            throttle_duration_sec=5.0
        )
    
    def generate_waypoints(self):
        """Generate waypoints based on trajectory type."""
        if self.trajectory_type == 'hover':
            return self.generate_hover_waypoints()
        elif self.trajectory_type == 'square':
            return self.generate_square_waypoints()
        elif self.trajectory_type == 'circle':
            return self.generate_circle_waypoints()
        elif self.trajectory_type == 'figure8':
            return self.generate_figure8_waypoints()
        elif self.trajectory_type == 'yaw_sweep':
            return self.generate_yaw_sweep_waypoints()
        elif self.trajectory_type == 'custom':
            return self.generate_custom_waypoints()
        else:
            self.get_logger().error(f"Unknown trajectory type: {self.trajectory_type}")
            return []
    
    def generate_hover_waypoints(self):
        """Generate single hover waypoint."""
        poses = []
        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = self.height
        pose.orientation.w = 1.0                            # Identity quaternion (yaw=0)
        poses.append(pose)
        return poses
    
    def generate_square_waypoints(self):
        """Generate square trajectory waypoints."""
        poses = []
        half_size = self.size / 2.0
        
        # Define square corners (counter-clockwise)
        corners = [
            (half_size, half_size, self.height, 0.0),      # Front-right
            (-half_size, half_size, self.height, 0.0),     # Front-left
            (-half_size, -half_size, self.height, 0.0),    # Back-left
            (half_size, -half_size, self.height, 0.0),     # Back-right
        ]
        
        for x, y, z, yaw in corners:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation = self.yaw_to_quaternion(yaw)
            poses.append(pose)
        
        return poses
    
    def generate_circle_waypoints(self):
        """Generate circular trajectory waypoints."""
        poses = []
        radius = self.size / 2.0
        
        for i in range(self.num_points):
            angle = 2 * math.pi * i / self.num_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = self.height
            pose.orientation = self.yaw_to_quaternion(angle)
            poses.append(pose)
        
        return poses
    
    def generate_figure8_waypoints(self):
        """Generate figure-8 trajectory waypoints."""
        poses = []
        radius = self.size / 4.0
        
        for i in range(self.num_points):
            t = 2 * math.pi * i / self.num_points
            x = radius * math.sin(t)
            y = radius * math.sin(2 * t) / 2
            
            # Calculate yaw to point along trajectory
            if i < self.num_points - 1:
                t_next = 2 * math.pi * (i + 1) / self.num_points
                x_next = radius * math.sin(t_next)
                y_next = radius * math.sin(2 * t_next) / 2
                yaw = math.atan2(y_next - y, x_next - x)
            else:
                yaw = 0.0
            
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = self.height
            pose.orientation = self.yaw_to_quaternion(yaw)
            poses.append(pose)
        
        return poses
    
    def generate_yaw_sweep_waypoints(self):
        """Generate waypoints that sweep through yaw angles at fixed position."""
        poses = []
        
        # Stay at origin, rotate through yaw angles
        yaw_angles = [0, math.pi/2, math.pi, -math.pi/2, 0]
        
        for yaw in yaw_angles:
            pose = Pose()
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = self.height
            pose.orientation = self.yaw_to_quaternion(yaw)
            poses.append(pose)
        
        return poses
    
    def generate_custom_waypoints(self):
        """Generate waypoints from custom parameter list."""
        poses = []
        
        # Custom waypoints should be [x1,y1,z1,yaw1, x2,y2,z2,yaw2, ...]
        if len(self.custom_waypoints) % 4 != 0:
            self.get_logger().error("Custom waypoints must have 4 values per point (x,y,z,yaw).")
            return []
        
        for i in range(0, len(self.custom_waypoints), 4):
            pose = Pose()
            pose.position.x = self.custom_waypoints[i]
            pose.position.y = self.custom_waypoints[i+1]
            pose.position.z = self.custom_waypoints[i+2]
            pose.orientation = self.yaw_to_quaternion(self.custom_waypoints[i+3])
            poses.append(pose)
        
        return poses
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion."""
        # Assuming roll=0, pitch=0
        qw = math.cos(yaw / 2.0)
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw / 2.0)
        
        # Return quaternion
        q = Pose().orientation
        q.x = qx
        q.y = qy
        q.z = qz
        q.w = qw
        return q
    
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
            
            elif name in ['waypoints_topic', 'frame_id']:
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason=f"{name} must be a string.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name == 'trajectory_type':
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason="trajectory_type must be a string.")
                valid_types = ['hover', 'square', 'circle', 'figure8', 'yaw_sweep', 'custom']
                if value not in valid_types:
                    return SetParametersResult(successful=False, reason=f"trajectory_type must be one of {valid_types}.")
                self.trajectory_type = value
                self.published = False  # Allow republishing with new trajectory
                self.get_logger().info(f"Updated trajectory_type: {value}")
            
            elif name == 'publish_once':
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason="publish_once must be a bool.")
                self.publish_once = value
                self.get_logger().info(f"Updated publish_once: {value}")
            
            elif name in ['height', 'size']:
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name, float(value))
                self.published = False  # Allow republishing with new parameters
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name == 'num_points':
                if not isinstance(value, int) or value <= 0:
                    return SetParametersResult(successful=False, reason="num_points must be > 0.")
                self.num_points = value
                self.published = False  # Allow republishing with new parameters
                self.get_logger().info(f"Updated num_points: {value}")
            
            elif name == 'custom_waypoints':
                if not isinstance(value, list):
                    return SetParametersResult(successful=False, reason="custom_waypoints must be a list.")
                self.custom_waypoints = value
                self.published = False  # Allow republishing with new waypoints
                self.get_logger().info(f"Updated custom_waypoints: {len(value)/4} points")
        
        return SetParametersResult(successful=True)
    
    def destroy_node(self):
        """Clean up before shutting down."""
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WaypointsPublisher()
    except Exception as e:
        print(f"[FATAL] WaypointsPublisher failed to initialize: {e}", file=sys.stderr)
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