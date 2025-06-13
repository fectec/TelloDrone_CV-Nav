# Vision-Based Localization and Pose Control for RoboMaster TT Tello Talent Drone Trajectory Execution using ChArUco Board

<p align="justify">This project is based on the work of Kenni Nilsson at the University of Southern Denmark, titled <a href="https://github.com/Kenil16/master_project" target="_blank">"Vision Based Navigation and Precision Landing of a Drone"</a>, developed as a master's project between fall 2020 and spring 2021 [1].</p>

<p align="justify">The objective is to enable the RoboMaster TT Tello Talent Drone to reach a final pose from an initial pose, using a ChArUco board as guidance.</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e05c8ab-ebcf-4278-ad66-456e38ac500d" alt="ROS2 DC Motor Control Layout" width="50%"/>
</p>

<p align="justify">As Nilsson mentions: "Using the Global Positioning System (GPS), the UAV can be programmed to fly between destinations. However, most of these systems present an error in the range of meters. To achieve greater precision, real-time kinematics (RTK) can be used, which reduces GPS error to centimeters. However, RTK is quite expensive and therefore not an optimal solution for low-cost applications. Additionally, for indoor navigation, the use of GPS would not be possible due to the reduction in signal intensity" [1].</p>

<p align="justify">A viable alternative consists of incorporating cameras into the UAV and processing the data through computer vision. By placing markers in the environment, it is possible to estimate the drone's pose with sufficient precision to allow autonomous flight tasks [1].</p>

<p align="justify">This project assumes that the UAV already operates with vision-based navigation, using a fixed ChArUco board on the ground as a spatial reference. This strategy offers an accessible and effective solution in contexts where GPS is not available or its precision is limited, such as indoors or with low-cost systems.</p>

<p align="justify">Once the drone's pose is estimated with respect to the ChArUco board, a PID controller will be implemented to guide it from an initial pose to a target pose, thus completing a trajectory.</p>

## Task Description

![Trajectory Task](https://github.com/user-attachments/assets/31ab928d-3ed6-4c3d-8e3d-951eb0811c3d)

<p align="justify">The ChArUco board used consists of a 9-column by 24-row grid. Each cell measures 0.1 meters per side and alternates between black squares and ArUco markers, which have a size of 0.08 meters. In total, the board has dimensions of 0.9 × 2.4 meters.</p>

<p align="justify">The DICT_4X4_250 dictionary is employed, which contains 250 unique identifiers encoded in a 4x4 bit pattern. Although this dictionary offers fewer IDs compared to more complex ones, its simplicity facilitates more robust and precise detection, especially under low resolution conditions or oblique angles.</p>

<p align="justify">The board's reference frame is located at its geometric center.</p>

<p align="justify">The UAV will be initially placed at a coordinate in the system defined by the board, from where it will take off. It will use the camera located on its lower part to capture images of the ChArUco board situated on the ground. From these images, the drone's pose (position and orientation) with respect to the board will be estimated in real time.</p>

<p align="justify">This estimation will be used by the control system to guide the drone from an initial pose to a target pose, fulfilling a trajectory.<p>

## Development

<p align="justify">The system was implemented in <a href="https://docs.ros.org/en/humble/Installation.html" target="_blank">ROS2 Humble</a>  using four main nodes that work together:</p>

### System Architecture

<ul>
 <li><b>tello_driver/tello_driver_node</b>: <p align="justify">Manages communication with the UAV through the <a href="https://github.com/damiafuentes/DJITelloPy" target="_blank">djitellopy</a> library. Transmits images from the lower camera and converts velocity commands into RC instructions for drone control.</p></li>
 
 <li><b>tello_vision/pose_estimator</b>: <p align="justify">Estimates the drone's pose through ChArUco board detection using OpenCV. Publishes the estimated pose for the controller and transforms (TF) for visualization in RViz.</p></li>
 
 <li><b>tello_control/waypoint_service</b>: <p align="justify">Service that provides waypoints one by one upon request, defining the trajectory that the drone must follow.</p></li>
 
 <li><b>tello_control/pose_controller</b>: <p align="justify">Controls the drone's movement based on current pose and target waypoints. Implements PID controllers for each degree of freedom and publishes velocity commands.</p></li>
</ul>

### System Flow

<p align="justify">The <code>tello_driver_node</code> publishes images from the lower camera on the <code>/downward/image_raw</code> topic and subscribes to <code>/cmd_vel</code> to convert <code>Twist</code> messages into drone RC commands.</p>

<p align="justify">The <code>pose_estimator</code> node processes these images by creating a ChArUco board with the specified characteristics and a corresponding detector. Pose estimation is performed through the following steps:</p>

<ul>
 <li><p align="justify">Detection of the ChArUco board in the captured image.</p></li>

 <li><p align="justify">Extraction of corners and correspondences between 3D board points and 2D image points.</p></li>

 <li><p align="justify">Centering of the board origin at its geometric center.</p></li>

 <li><p align="justify">Resolution of the PnP (Perspective-n-Point) problem using <code>cv2.solvePnP</code>.</p></li>

 <li><p align="justify">Conversion to transformation matrix and inversion to obtain camera pose relative to the board.</p></li>

 <li><p align="justify">Correction of the camera transform to obtain the final drone pose.</p></li>
</ul>

<p align="justify">This pose is published on the <code>/drone/pose</code> topic and corresponding TF transforms are sent for visualization. The node publishes two transforms: a static one between world and board, and a dynamic one between board and drone.</p>

<p align="justify">The <code>waypoint_service</code> node defines and provides the sequence of target poses through the <code>/get_next_waypoint</code> service trigger.</p>

<p align="justify">Finally, the <code>pose_controller</code> node implements trajectory control:</p>

<ul>
 <li><p align="justify">Calculates position (x, y, z) and orientation (yaw) errors between current pose and target waypoint.</p></li>

 <li><p align="justify">Transforms errors to the drone's local reference frame.</p></li>

 <li><p align="justify">Applies independent PID controllers for each degree of freedom.</p></li>

 <li><p align="justify">Publishes control commands as Twist messages.</p></li>

 <li><p align="justify">Evaluates if the waypoint has been reached by comparing position and orientation error with predefined thresholds.</p></li>

 <li><p align="justify">Advances to the next waypoint when convergence criteria are met.</p></li>

 <li><p align="justify">The system considers that a waypoint has been reached when position and orientation errors are below established tolerances and control outputs remain stable.</p></li>
</ul>

## Results

<p align="justify">To evaluate the system's performance, a navigation test was conducted where the drone had to move from an initial pose to a target waypoint. The initial pose was established at the bottom-right corner (x = -0.45, y = 1.2, z = 0.0, yaw = 0.0) of the ChArUco board, while the target waypoint corresponded to the top-left corner (x = 0.45, y = -1.2, z = 5.0, yaw = 0.0).</p>

### Controller Precision

<p align="justify">The controller reached the destination, with the following residual errors:</p>

<ul>
 <li>X position error: 0.2m</li>
 <li>Y position error: 0.01m</li>
 <li>Z position error: 0.1m</li>
</ul>

<p align="justify">The system did not achieve completely exact convergence to the target waypoint. A possible cause of this limitation lies in the fact that the constant movement of the drone during flight introduces disturbances in pose estimation. The natural vibrations of the UAV and the inherent oscillations of the control system affect the quality of captured images, which directly impacts the precision of ChArUco board detection.</p>

## Usage

<p align="justify">You can tune parameters in <code>tello_bringup/config/pose_trajectory_config.yaml</code>. Most importantly for your own application:</p>

<p align="justify">Tello's camera calibration parameters (<code>camera_matrix</code>, <code>distortion_coeffs</code>), ChArUco board characteristics (<code>board_squares_x</code>, <code>board_squares_y</code>, <code>square_length</code>, <code>marker_length</code>, <code>aruco_dict</code>), custom waypoints (<code>custom_waypoints</code>), PID constants for each DOF (<code>pid_x_kp</code>, <code>pid_x_ki</code>, <code>pid_x_kd</code>, <code>pid_y_kp</code>, <code>pid_y_ki</code>, <code>pid_y_kd</code>, <code>pid_z_kp</code>, <code>pid_z_ki</code>, <code>pid_z_kd</code>, <code>pid_yaw_kp</code>, <code>pid_yaw_ki</code>, <code>pid_yaw_kd</code>), tolerances (<code>position_tolerance</code>, <code>yaw_tolerance</code>) </p>

<p align="justify">Place a ChArUco board on a flat surface</p>

<p align="justify">Launch the system by running:</p>

```bash
ros2 launch tello_bringup pose_trajectory.launch.py
```

<p align="justify">When ready, take off by executing:</p>

```bash
ros2 topic pub /tello/takeoff std_msgs/msg/Empty "{}"
```

## References

<p align="justify">[1] K. Nilsson, Vision Based Navigation and Precision Landing of a Drone, Master’s project, Faculty of Engineering, University of Southern Denmark, Odense, Denmark, Sept. 2020–Jun. 2021.</p>
