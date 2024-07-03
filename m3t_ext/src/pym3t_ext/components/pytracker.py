# Standard libraries
from time import time

# Third-party libraries
import cv2
from tabulate import tabulate
import numpy as np

# Custom libraries
import pym3t


class PyTracker(pym3t.Tracker):
    """
    Extension of the pym3t.Tracker class.
    """
    def run_tracker_process(
        self,
        execute_detection_and_start_tracking: bool = True,
        nb_images: int = 100,
        display_images: bool = False,
        stop_at_each_image: bool = False,
        display_timings: bool = True,
    ) -> bool:
        """Run the tracking process for a given number of iterations.
        Re-implementation of the pym3t.Tracker::RunTrackerProcess method, but
        not an override.

        Args:
            execute_detection_and_start_tracking (bool, optional): Whether to execute
                the detection step and start the tracking process. Defaults to True.
            nb_images (int, optional): Number of images to process. Defaults to 100.
            stop_at_each_image (bool, optional): Whether to stop at each image. Defaults
                to False.
            display_timings (bool, optional): Whether to display the timings of each
                step of the tracking process at the end of the iterations. Defaults to
                True.

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
        
        # Tracking process
        for i in range(nb_images):
            
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
