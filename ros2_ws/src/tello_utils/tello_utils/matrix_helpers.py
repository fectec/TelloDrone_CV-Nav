"""
Utilities to “reverse” the order of Euler‐angle rotations in a 4×4 homogeneous transform.

Functions:
    - reverse_xyz_to_zyx_4x4(T): given T = [Rz(γ)·Ry(β)·Rx(α), t; 0,1], return [Rx(α)·Ry(β)·Rz(γ), t; 0,1]
    - reverse_zyx_to_xyz_4x4(T): given T = [Rx(α)·Ry(β)·Rz(γ), t; 0,1], return [Rz(γ)·Ry(β)·Rx(α), t; 0,1]

Internally:
    • Extract the top‐left 3×3 rotation block R.
    • Decompose that R into Euler angles (α, β, γ) according to the assumed convention.
    • Re‐compose in the opposite order.
    • Re‐insert the unchanged translation t into the new 4×4.
"""

import numpy as np
import math
import cv2

def Rx(alpha: float) -> np.ndarray:
    """Rotation about X by angle (radians)."""
    c = math.cos(alpha)
    s = math.sin(alpha)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ], dtype=np.float64)

def Ry(beta: float) -> np.ndarray:
    """Rotation about Y by angle (radians)."""
    c = math.cos(beta)
    s = math.sin(beta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float64)

def Rz(gamma: float) -> np.ndarray:
    """Rotation about Z by angle (radians)."""
    c = math.cos(gamma)
    s = math.sin(gamma)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ], dtype=np.float64)

def extract_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """
    Given a 3×3 rotation R assumed to be R = Rz(γ)·Ry(β)·Rx(α) (intrinsic XYZ order),
    return (α, β, γ) in radians.
    """
    if R.shape != (3, 3):
        raise ValueError(f"Extract_euler_xyz: expected a 3×3 matrix, got shape {R.shape}.")

    # sy = sqrt(R[0,0]^2 + R[1,0]^2)
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = (sy < 1e-6)

    if not singular:
        alpha = math.atan2(R[2, 1], R[2, 2])   # roll about X
        beta  = math.atan2(-R[2, 0], sy)       # pitch about Y
        gamma = math.atan2(R[1, 0], R[0, 0])   # yaw   about Z
    else:
        # Gimbal lock: β ~ ±90°, set γ = 0
        alpha = math.atan2(-R[1, 2], R[1, 1])
        beta  = math.atan2(-R[2, 0], sy)
        gamma = 0.0

    return alpha, beta, gamma

def extract_euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """
    Given a 3×3 rotation R assumed to be R = Rx(α)·Ry(β)·Rz(γ) (intrinsic ZYX order),
    return (α, β, γ) in radians.
    """
    if R.shape != (3, 3):
        raise ValueError(f"Extract_euler_zyx: expected a 3×3 matrix, got shape {R.shape}.")

    # In this convention, R[0,2] = sin(β)
    beta = math.asin(np.clip(R[0, 2], -1.0, 1.0))
    cb = math.cos(beta)

    if abs(cb) > 1e-6:
        alpha = math.atan2(-R[1, 2] / cb, R[2, 2] / cb)   # α from R[1,2] & R[2,2]
        gamma = math.atan2(-R[0, 1] / cb, R[0, 0] / cb)   # γ from R[0,1] & R[0,0]
    else:
        # Gimbal lock: set γ = 0
        alpha = math.atan2(-R[1, 2], R[1, 1])
        gamma = 0.0

    return alpha, beta, gamma

def reverse_xyz_to_zyx_4x4(T: np.ndarray) -> np.ndarray:
    """
    Assume T is a 4×4 homogeneous transform whose rotation block R = Rz(γ)·Ry(β)·Rx(α)
    (intrinsic XYZ build). Extract (α, β, γ) and return a new 4×4:
      [Rx(α)·Ry(β)·Rz(γ),  t;  0, 1]
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Reverse_xyz_to_zyx_4x4: expected a 4×4 matrix, got {T.shape}.")

    R = T[:3, :3]
    t = T[:3,  3].copy()

    # 1) Extract (α, β, γ) assuming R = Rz·Ry·Rx
    alpha, beta, gamma = extract_euler_xyz(R)

    # 2) Re-compose in opposite (ZYX) order: R' = Rx(α)·Ry(β)·Rz(γ)
    R_rev = Rx(alpha) @ Ry(beta) @ Rz(gamma)

    # 3) Build new 4×4
    T_rev = np.eye(4, dtype=np.float64)
    T_rev[:3, :3] = R_rev
    T_rev[:3,  3] = t
    return T_rev

def reverse_zyx_to_xyz_4x4(T: np.ndarray) -> np.ndarray:
    """
    Assume T is a 4×4 homogeneous transform whose rotation block R = Rx(α)·Ry(β)·Rz(γ)
    (intrinsic ZYX build). Extract (α, β, γ) and return a new 4×4:
      [Rz(γ)·Ry(β)·Rx(α),  t;  0, 1]
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Reverse_zyx_to_xyz_4x4: expected a 4×4 matrix, got {T.shape}.")

    R = T[:3, :3]
    t = T[:3,  3].copy()

    # 1) Extract (α, β, γ) assuming R = Rx·Ry·Rz
    alpha, beta, gamma = extract_euler_zyx(R)

    # 2) Re-compose in opposite (XYZ) order: R' = Rz(γ)·Ry(β)·Rx(α)
    R_rev = Rz(gamma) @ Ry(beta) @ Rx(alpha)

    # 3) Build new 4×4
    T_rev = np.eye(4, dtype=np.float64)
    T_rev[:3, :3] = R_rev
    T_rev[:3,  3] = t
    return T_rev

def vecs_to_matrix(rvec, tvec):
    """Convert rvec, tvec to a 4x4 transformation matrix."""
    rvec = np.asarray(rvec, dtype=np.float32)
    tvec = np.asarray(tvec, dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def matrix_to_vecs(T):
    """Convert a 4x4 transformation matrix to rvec, tvec."""
    R = T[:3, :3]
    tvec = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten(), tvec.flatten()

# GLOBALS
Rz90    = Rz(np.pi/2)   # Precomputed rotation matrix for 90 degrees around Z-axis
Rz180   = Rz(np.pi)     # Precomputed rotation matrix for 180 degrees around Z-axis
Rz270   = Rz(3*np.pi/2) # Precomputed rotation matrix for 270 degrees around Z-axis
Rx90    = Rx(np.pi/2)   # Precomputed rotation matrix for 90 degrees around X-axis
Rx180   = Rx(np.pi)     # Precomputed rotation matrix for 180 degrees around X-axis
Rx270   = Rx(3*np.pi/2) # Precomputed rotation matrix for 270 degrees around X-axis
Ry90    = Ry(np.pi/2)   # Precomputed rotation matrix for 90 degrees around Y-axis
Ry180   = Ry(np.pi)     # Precomputed rotation matrix for 180 degrees around Y-axis
Ry270   = Ry(3*np.pi/2) # Precomputed rotation matrix for 270 degrees around Y-axis