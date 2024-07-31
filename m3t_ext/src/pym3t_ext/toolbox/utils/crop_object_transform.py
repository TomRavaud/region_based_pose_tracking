# Standard libraries
from typing import Tuple

# Third-party libraries
import numpy as np
# from torchvision.transforms.functional import crop, resize
import torch
from torchvision.transforms.functional import resize


class CropObjectTransform:
    
    def __init__(
        self,
        resize: tuple = (480, 640),
        scale_factor: bool = True,
    ) -> None:
        """Constructor.

        Args:
            resize (Resolution, optional): Target aspect ratio (height, width).
                Defaults to (480, 640).
            focus_on_object (bool, optional): If True, the object is centered in
                the cropped image. Defaults to False.

        Raises:
            ValueError: If the width is less than the height.
        """
        if resize[1] < resize[0]:
            raise ValueError("The width must be greater than the height.")
        
        self._resize = resize
        self._aspect = max(resize) / min(resize)
        self._scale_factor = scale_factor
        
    def __call__(
        self,
        image: torch.Tensor,
        bbox: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        
        h, w = image.shape[1:3]
        
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]

        # Compute the size and center of the bounding box
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2

        # Modify the size of the bounding box
        new_bbox_width = bbox_width * (1 + self._scale_factor)
        new_bbox_height = bbox_height * (1 + self._scale_factor)

        # Ensure that the new bounding box has the correct aspect ratio
        if new_bbox_width / new_bbox_height > self._aspect:
            new_bbox_height = new_bbox_width / self._aspect
        else:
            new_bbox_width = new_bbox_height * self._aspect

        # Compute the new bounding box coordinates
        crop_x_min = bbox_center_x - new_bbox_width / 2
        crop_y_min = bbox_center_y - new_bbox_height / 2
        crop_x_max = bbox_center_x + new_bbox_width / 2
        crop_y_max = bbox_center_y + new_bbox_height / 2

        # Ensure that the crop is inside the image; if not, shift the crop
        if crop_x_min < 0:
            crop_x_min = 0
            crop_x_max = new_bbox_width
        if crop_x_max > w:
            crop_x_max = w
            crop_x_min = w - new_bbox_width
        if crop_y_min < 0:
            crop_y_min = 0
            crop_y_max = new_bbox_height
        if crop_y_max > h:
            crop_y_max = h
            crop_y_min = h - new_bbox_height

        crop_x_min = int(crop_x_min)
        crop_y_min = int(crop_y_min)
        crop_x_max = int(crop_x_max)
        crop_y_max = int(crop_y_max)

        # Compute the new bounding box coordinates
        new_bbox_x_min = x_min - crop_x_min
        new_bbox_y_min = y_min - crop_y_min
        new_bbox_x_max = x_max - crop_x_min
        new_bbox_y_max = y_max - crop_y_min
        new_bbox = np.array([[new_bbox_x_min, new_bbox_y_min], [new_bbox_x_max, new_bbox_y_max]])

        # Extract the crop and update the RGB images
        crop = image[:, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # Resize the crop
        crop = resize(crop, self._resize, antialias=True)
        
        # Update the bounding box
        new_bbox = new_bbox * (self._resize[1] / new_bbox_width)
        
        return crop, new_bbox
    
    def point_transform(
        self,
        points: np.ndarray,
        orig_size: tuple[int, int],
        valid_borders: bool = False,
    ) -> np.ndarray:
        """Transform points from the original image to the cropped and resized
        image.

        Args:
            points (np.ndarray): Points in the original image. Shape (N, 2).
            orig_size (tuple[int, int]): Original image size (height, width).
            valid_borders (bool, optional): If False, points outside the borders
                are set to -1, otherwise, they are set to the top or bottom border.
                Defaults to False.

        Returns:
            np.ndarray: Points' coordinates in the cropped and resized image.
        """
        # Original shape and aspect ratio
        height, width = orig_size
        aspect_ratio_original = width / height
        
        # Target shape and aspect ratio
        height_target, width_target = self._resize
        aspect_ratio_target = width_target / height_target
        
        # Image top and bottom will be cropped
        if aspect_ratio_target >= aspect_ratio_original:
            
            # Compute the new height to match the target aspect ratio
            height_tmp = width / aspect_ratio_target
            
            # Size of the borders to remove
            border = (height - height_tmp) / 2
            
            # Check if the points are inside the borders
            is_outside = (points[:, 1] < border) | (points[:, 1] > height - border)
            
            # Compute the resize factor
            resize_factor = width_target / width
            
            # Compute the new coordinates
            points_new = np.zeros_like(points, dtype=np.int32)
            points_new[~is_outside, 0] = points[~is_outside, 0] * resize_factor
            points_new[~is_outside, 1] =\
                (points[~is_outside, 1] - border) * resize_factor
            
            if valid_borders:
                points_new[is_outside, 0] = points[is_outside, 0] * resize_factor
                # If the point is above the top border, set it to the top border and
                # if it is below the bottom border, set it to the bottom border
                points_new[is_outside, 1] = np.where(
                    points[is_outside, 1] < border, 0, height_target - 1
                )
            else:
                points_new[is_outside] = -1
            
            return points_new
        
        else:
            return NotImplementedError(
                "The case where the target aspect ratio is smaller than the "
                "original aspect ratio is not implemented."
            )
    

if __name__ == "__main__":
    pass
