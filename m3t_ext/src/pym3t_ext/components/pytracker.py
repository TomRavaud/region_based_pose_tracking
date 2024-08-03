# Standard libraries
from time import time
from pathlib import Path

# Third-party libraries
import cv2
from tabulate import tabulate
import numpy as np
from tqdm import tqdm

# Custom libraries
import pym3t
import pym3t_ext.toolbox.geometry.metrics as custom_metrics


class PyTracker(pym3t.Tracker):
    """
    Extension of the pym3t.Tracker class.
    """
    def run_tracker_process(
        self,
        model: str = "model",
        scene: str = "scene",
        execute_detection_and_start_tracking: bool = True,
        nb_images: int = 100,
        display_images: bool = False,
        stop_at_each_image: bool = False,
        display_timings: bool = True,
        body2world_poses_gt: np.ndarray = None,
        body: pym3t.Body = None,
        reset_pose: dict = None,
        log_dir: Path = None,
    ) -> bool:
        """Run the tracking process for a given number of iterations.
        Re-implementation of the pym3t.Tracker::RunTrackerProcess method, but
        not an override.

        Args:
            model (str, optional): Model name. Defaults to "model".
            scene (str, optional): Scene name. Defaults to "scene".
            execute_detection_and_start_tracking (bool, optional): Whether to execute
                the detection step and start the tracking process. Defaults to True.
            nb_images (int, optional): Number of images to process. Defaults to 100.
            display_images (bool, optional): Whether to display the images. Defaults
                to False.
            stop_at_each_image (bool, optional): Whether to stop at each image. Defaults
                to False.
            display_timings (bool, optional): Whether to display the timings of each
                step of the tracking process at the end of the iterations. Defaults to
                True.
            body2world_poses_gt (np.ndarray, optional): Ground truth poses of the object
                in the world frame. Defaults to None.
            body (pym3t.Body, optional): Body object. Defaults to None.
            reset_pose (dict, optional): Dictionary containing the reset pose criterion.
                Defaults to None.
            log_dir (Path, optional): Directory to save the scores. Defaults to None.

        Returns:
            bool: Whether the tracking process was successful.
        """
        # Check if tracker is set up
        if not self.set_up:
            print(f"Set up tracker {self.name} first.")
            return False
        
        # Initialize objects' states (detecting, starting or tracking)
        if execute_detection_and_start_tracking:
            self.ExecuteDetection(start_tracking=True)
        
        # Initialize a variable to store the number of iterations effectively done
        nb_images_done = nb_images
        
        # Initialize timings
        timings = {
            "UpdateCameras": 0.0,
            "UpdateSubscribers": 0.0,
            "CalculateConsistentPoses": 0.0,
            "ExecuteDetectingStep": 0.0,
            "ExecuteStartingStep": 0.0,
            "ExecuteTrackingStep": 0.0,
            "UpdatePublishers": 0.0,
            "UpdateViewers": 0.0,
            "Total": 0.0,
        }
        
        if log_dir is not None:
            # Create a text file to store the poses
            with open(log_dir / f"poses_{model}_{scene}.txt", "w") as f:
                f.write("r11\tr12\tr13\tr21\tr22\tr23\tr31\tr32\tr33\ttx\tty\ttz\n")
                if body2world_poses_gt is not None:
                # Write the initial GT pose
                    pose_gt = body2world_poses_gt[0]
                    f.write(
                        f"{pose_gt[0, 0]}\t{pose_gt[0, 1]}\t{pose_gt[0, 2]}\t"
                        f"{pose_gt[1, 0]}\t{pose_gt[1, 1]}\t{pose_gt[1, 2]}\t"
                        f"{pose_gt[2, 0]}\t{pose_gt[2, 1]}\t{pose_gt[2, 2]}\t"
                        f"{pose_gt[0, 3]}\t{pose_gt[1, 3]}\t{pose_gt[2, 3]}\n"
                    )
        
        # Define the metric
        if reset_pose is not None and reset_pose.do_reset:
            is_metric_defined = False
            criterion = reset_pose.criterion
            metric_class_name = list(criterion.keys())[0]
            # Get the metric class
            try:
                metric_class = getattr(custom_metrics, metric_class_name)
            except AttributeError:
                raise ValueError(f"Unknown metric: {metric_class_name}")
            # Instantiate the metric by passing the parameters
            try:
                my_metric = metric_class(**criterion[metric_class_name])
            except TypeError as e:
                print(f"Error instantiating metric {metric_class_name}: {e}")
            else:
                is_metric_defined = True
        
        
        # Tracking process
        for i in tqdm(range(nb_images)):
            
            t_start_i = time()
            
            # Update all Cameras
            if not self.UpdateCameras(i):
                return False
            t_update_cameras_i = time()
            timings["UpdateCameras"] += t_update_cameras_i - t_start_i
            
            # Update all Subscribers
            if not self.UpdateSubscribers(i):
                return False
            t_update_subscribers_i = time()
            timings["UpdateSubscribers"] += t_update_subscribers_i - t_update_cameras_i
            
            # Calculate consistent Link and Body poses
            # (necessary if subscribers have modified some link poses)
            if not self.CalculateConsistentPoses():
                return False
            t_calculate_consistent_poses_i = time()
            timings["CalculateConsistentPoses"] +=\
                t_calculate_consistent_poses_i - t_update_subscribers_i
            
            # Run Detecting step
            if not self.ExecuteDetectingStep(i):
                return False
            t_execute_detecting_step_i = time()
            timings["ExecuteDetectingStep"] +=\
                t_execute_detecting_step_i - t_calculate_consistent_poses_i
            
            # Run Starting step
            if not self.ExecuteStartingStep(i):
                return False
            t_execute_starting_step_i = time()
            timings["ExecuteStartingStep"] +=\
                t_execute_starting_step_i - t_execute_detecting_step_i

            # Run Tracking step
            if not self.ExecuteTrackingStep(i):
                return False
            t_execute_tracking_step_i = time()
            timings["ExecuteTrackingStep"] +=\
                t_execute_tracking_step_i - t_execute_starting_step_i

            # Update all Publishers
            if not self.UpdatePublishers(i):
                return False
            t_update_publishers_i = time()
            timings["UpdatePublishers"] +=\
                t_update_publishers_i - t_execute_tracking_step_i
            
            if display_images:
                # Update all Viewers
                if not self.UpdateViewers(i):
                    return False
                timings["UpdateViewers"] += time() - t_update_publishers_i
            
            timings["Total"] += time() - t_start_i
            
            if display_images and stop_at_each_image:
                k = cv2.waitKey(0)
                if k == ord("q"):
                    nb_images_done = i + 1
                    break
            
            
            if body is not None:
                # Get the object pose relative to the camera frame
                pose_refined = body.body2world_pose.matrix.copy()
            
            if log_dir is not None:
                # Save the pose
                with open(log_dir / f"poses_{model}_{scene}.txt", "a") as f:
                    f.write(
                        f"{pose_refined[0, 0]}\t{pose_refined[0, 1]}\t"
                        f"{pose_refined[0, 2]}\t{pose_refined[1, 0]}\t"
                        f"{pose_refined[1, 1]}\t{pose_refined[1, 2]}\t"
                        f"{pose_refined[2, 0]}\t{pose_refined[2, 1]}\t"
                        f"{pose_refined[2, 2]}\t{pose_refined[0, 3]}\t"
                        f"{pose_refined[1, 3]}\t{pose_refined[2, 3]}\n"
                    )
            
            # Reset pose if needed
            if body is not None and body2world_poses_gt is not None:
                # Get the GT pose
                pose_gt = body2world_poses_gt[i+1]
                
                # Reset the body2world pose to the GT pose if the tracking is lost
                if reset_pose is not None\
                    and reset_pose.do_reset\
                        and is_metric_defined:
                    
                    # Compute the metric
                    success = my_metric(T_gt=pose_gt, T_est=pose_refined)
                    
                    # Tracking is lost
                    if not success:
                        # Reset the body2world pose to the GT pose
                        body.body2world_pose = pym3t.Transform3fA(
                            np.array(pose_gt, dtype=np.float32)
                        )
                        # Restart the modalities
                        for modality in self.modalities:
                            if type(modality).__name__ in [
                                "DeepRegionModality",
                                "DeepCLinesRegionModality",
                            ]:
                                modality.start_modality(0, 0)
                            else:
                                modality.StartModality(0, 0)
                        
        # Display timings
        if nb_images_done > 0 and display_timings:

            # Prepare data for the table
            table_data = []
            for step, duration in timings.items():
                rate = nb_images_done / duration if duration > 0 else np.nan
                percentage = duration / timings['Total'] * 100\
                    if timings['Total'] > 0 else np.nan
                table_data.append([step, f"{rate:.1f}", f"{percentage:.1f}"])

            # Create the table and display it
            output = tabulate(
                table_data,
                headers=["Step", "Rate (Hz)", "% Total"],
                tablefmt="fancy_grid",
                numalign="center",
            )
            print(output)
        
        return True
