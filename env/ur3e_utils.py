import time
import numpy as np


class Rate:
    def __init__(self, control_hz, slack_time=0.001):
        self.control_hz = control_hz
        self.slack_time = slack_time

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t_curr = time.time()
        t_end = self.t_start + 1 / self.control_hz
        t_wait = t_end - t_curr
        if t_wait > 0:
            t_sleep = t_wait - self.slack_time
            if t_sleep > 0:
                time.sleep(t_sleep)
            while time.time() < t_end:
                pass


def rotation_vector_to_rpy(rotation_vector):
    theta = np.linalg.norm(rotation_vector)
    if theta < 1e-10:  # Check for zero rotation
        return np.array([0.0, 0.0, 0.0])
    
    u = rotation_vector / theta  # Normalize the rotation vector
    K = np.array([[0, -u[2], u[1]],
                  [u[2], 0, -u[0]],
                  [-u[1], u[0], 0]])
    
    # Compute rotation matrix
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    # Extract roll, pitch, yaw
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    return np.array([roll, pitch, yaw])