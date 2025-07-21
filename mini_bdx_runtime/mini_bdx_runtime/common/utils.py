import jax.numpy as jp
import jax


class LowPassActionFilter:
    def __init__(self, control_freq, cutoff_frequency=30.0):
        self.last_action = 0
        self.current_action = 0
        self.control_freq = float(control_freq)
        self.cutoff_frequency = float(cutoff_frequency)
        self.alpha = self.compute_alpha()

    def compute_alpha(self):
        return (1.0 / self.cutoff_frequency) / (
            1.0 / self.control_freq + 1.0 / self.cutoff_frequency
        )

    def push(self, action: jax.Array) -> None:
        self.current_action = jp.array(action)

    def get_filtered_action(self) -> jax.Array:
        self.last_action = (
            self.alpha * self.last_action + (1 - self.alpha) * self.current_action
        )
        return self.last_action
    
def convert_wxyz_to_xyzw(q):
    """Convert quaternion from wxyz format to xyzw format."""
    # MuJoCo format is [w, x, y, z]
    # Reference format is [x, y, z, w]
    return jp.array([q[1], q[2], q[3], q[0]])

def quat_mul(q1, q2):
    """Multiply two quaternions in xyzw format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return jp.array([x, y, z, w])

def quat_diff(a: jp.ndarray, b: jp.ndarray) -> jp.ndarray:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    return 2.0 * jp.arcsin(
        jp.clip(
            jp.linalg.norm(
                mul[0:3],
                axis=-1), a_max=1.0)
    )

def quaternion_angle_diff(q1, q2):
    """Calculate the angle between two quaternions representing orientations"""
    # Ensure unit quaternions
    q1 = q1 / jp.linalg.norm(q1)
    q2 = q2 / jp.linalg.norm(q2)
    
    # Compute the dot product of the full quaternions
    dot_product = jp.sum(q1 * q2)
    
    # The absolute value is necessary because q and -q represent the same rotation
    dot_product = jp.abs(dot_product)
    
    # Clamp to valid domain for arccos
    dot_product = jp.clip(dot_product, -1.0, 1.0)
    
    # Calculate the angle
    angle = 2.0 * jp.arccos(dot_product)
    
    return angle

def calc_heading_quat(q):
    """Calculate quaternion representing only the heading (yaw) component."""
    # Quaternion format is [x, y, z, w]
    # Extract yaw angle
    sin_theta = 2.0 * (q[3] * q[2] - q[0] * q[1])  # w*z - x*y
    cos_theta = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])  # 1 - 2*(y*y + z*z)
    
    # Create a new quaternion with only yaw rotation (around Z axis)
    yaw = jp.arctan2(sin_theta, cos_theta)
    heading_q = jp.array([0.0, 0.0, jp.sin(0.5 * yaw), jp.cos(0.5 * yaw)])
    
    return heading_q

def quat_conjugate(q):
    """Calculate the conjugate of a quaternion (for unit quaternions, this is the inverse)."""
    # Quaternion format is [x, y, z, w]
    return jp.array([-q[0], -q[1], -q[2], q[3]])

def quat_rotate(q, v):
    """Rotate a vector v by quaternion q."""
    # Quaternion format is [x, y, z, w]
    u = q[:3]  # Vector part
    s = q[3]   # Scalar part
    
    # Formula: v' = v + 2s(u×v) + 2(u×(u×v))
    return v + 2.0 * jp.cross(u, jp.cross(u, v) + s * v)

def quat_from_euler_xyz(roll, pitch, yaw):
    cy = jp.cos(yaw * 0.5)
    sy = jp.sin(yaw * 0.5)
    cr = jp.cos(roll * 0.5)
    sr = jp.sin(roll * 0.5)
    cp = jp.cos(pitch * 0.5)
    sp = jp.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return jp.stack([qx, qy, qz, qw], axis=-1)