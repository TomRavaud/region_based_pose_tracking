import numpy as np


def cm_degree_score(T_gt: np.ndarray,
                    T_est: np.ndarray,
                    threshold_trans: float=5.,
                    threshold_rot: float=5.) -> tuple[float, float, bool]:
    """Compute the translation and rotation errors between two poses and check
    if they are below a given threshold.

    Args:
        T_gt (np.ndarray): Ground truth pose.
        T_est (np.ndarray): Estimated pose.
        threshold_trans (float, optional): Threshold on translation error
        (in cm). Defaults to 5..
        threshold_rot (float, optional): Threshold on rotation error
        (in degree). Defaults to 5.

    Returns:
        tuple[float, float, bool]: translation error, rotation error, success.
    """
    # Get rotation matrices and translation vectors
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    
    # Translation error
    error_trans = np.linalg.norm(t_gt - t_est)
    
    # Rotation error
    arg = (np.trace(np.dot(R_est.T, R_gt)) - 1) / 2
    # Check if the argument is out of bounds (due to numerical errors)
    if arg > 1:
        error_rot = 0
    elif arg < -1:
        error_rot = np.pi
    # Compute the error
    else:
        error_rot = np.arccos((np.trace(np.dot(R_est.T, R_gt)) - 1) / 2)
    
    # Convert the translation threshold to meters
    threshold_trans *= 1e-2
    
    # Convert the rotation threshold to radians
    threshold_rot *= np.pi / 180
    
    # Check if the error is below the threshold
    success = error_trans < threshold_trans and error_rot < threshold_rot
    
    return success
