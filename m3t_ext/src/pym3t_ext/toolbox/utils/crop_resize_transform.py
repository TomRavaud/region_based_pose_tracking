# Standard libraries
from typing import Tuple

# Third-party libraries
import numpy as np
import torchvision.transforms.functional as F
import torch


class CropResizeToAspectTransform:
    """
    Crop and resize the RGB observations to a target aspect ratio.
    """
    def __init__(self, resize: tuple = (480, 640), p: float = 1.0) -> None:
        """Constructor.

        Args:
            resize (Resolution, optional): Target aspect ratio (height, width).
                Defaults to (480, 640).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.

        Raises:
            ValueError: If the width is less than the height.
        """
        self._p = p
        
        if resize[1] < resize[0]:
            raise ValueError("The width must be greater than the height.")
        
        self._resize = resize
        self._aspect = max(resize) / min(resize)
    
    def __call__(
        self,
        image: torch.Tensor,
        bbox: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Crop and resize the image and bounding box to a target aspect ratio.

        Args:
            image (torch.Tensor): Image to crop and resize.
            bbox (np.ndarray): Bounding box coordinates. Shape (2, 2).

        Returns:
            Tuple[torch.Tensor, np.ndarray]: Cropped and resized image and
                bounding box.
        """
        h, w = image.shape[1:3]

        # Skip if the image is already at the target size
        if (h, w) == self._resize:
            return image, bbox

        # Match the width on input image with an image of target aspect ratio.
        if not np.isclose(w / h, self._aspect):
            r = self._aspect
            crop_h = w * 1 / r
            x0, y0 = w / 2, h / 2
            crop_box_size = (crop_h, w)
            crop_h, crop_w = min(crop_box_size), max(crop_box_size)
            x1, y1, x2, y2 = (
                x0 - crop_w / 2,
                y0 - crop_h / 2,
                x0 + crop_w / 2,
                y0 + crop_h / 2,
            )
            x1, y1, x2, y2 = tuple(map(int, (x1, y1, x2, y2)))
            
            # Crop the RGB images
            crop = image[:, y1:y2, x1:x2]
        else:
            crop = image
        
        # Resize the crop
        crop = F.resize(crop, self._resize, antialias=True)
        
        # Compute the new bounding box coordinates
        x_bbox_1, y_bbox_1 = bbox[0]
        x_bbox_2, y_bbox_2 = bbox[1]
        x_bbox_1_new, y_bbox_1_new = x_bbox_1 - x1, y_bbox_1 - y1
        x_bbox_2_new, y_bbox_2_new = x_bbox_2 - x1, y_bbox_2 - y1
        new_bbox = np.array([
            [x_bbox_1_new, y_bbox_1_new],
            [x_bbox_2_new, y_bbox_2_new],
        ])
        # Resizing
        new_bbox = new_bbox * (self._resize[1] / crop_w)
        
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


class CropResizeToObjectTransform:
    
    def __init__(
        self,
        resize: tuple = (480, 640),
        scale_factor: float = 1.5,
    ) -> None:
        """Constructor.

        Args:
            resize (Resolution, optional): Target aspect ratio (height, width).
                Defaults to (480, 640).
            scale_factor (float, optional): Scale factor to expand the bounding box.
                Defaults to 1.5.

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
        """Crop and resize an image around a bounding box.

        Args:
            image (torch.Tensor): Image to crop and resize.
            bbox (np.ndarray): Bounding box coordinates. Shape (2, 2).

        Returns:
            Tuple[torch.Tensor, np.ndarray]: Cropped and resized image and bounding box.
        """
        h, w = image.shape[1:3]
        
        x_bbox_1, y_bbox_1 = bbox[0]
        x_bbox_2, y_bbox_2 = bbox[1]

        # Compute the size and center of the bounding box
        bbox_w, bbox_h = x_bbox_2 - x_bbox_1, y_bbox_2 - y_bbox_1
        x_center, y_center = (x_bbox_1 + x_bbox_2) / 2, (y_bbox_1 + y_bbox_2) / 2

        # Expand the bounding box by the scale factor
        bbox_w, bbox_h = bbox_w * self._scale_factor, bbox_h * self._scale_factor

        # Ensure the box has the correct aspect ratio
        if bbox_w / bbox_h > self._aspect:
            bbox_h = bbox_w / self._aspect
        else:
            bbox_w = bbox_h * self._aspect

        # Update the box coordinates
        x1, y1, x2, y2 = (
            x_center - bbox_w / 2,
            y_center - bbox_h / 2,
            x_center + bbox_w / 2,
            y_center + bbox_h / 2,
        )

        # Ensure the box is within image bounds; if not, shift it
        if x1 < 0:
            x1, x2 = 0, bbox_w
        if y1 < 0:
            y1, y2 = 0, bbox_h
        if x2 > w:
            x1, x2 = w - bbox_w, w
        if y2 > h:
            y1, y2 = h - bbox_h, h

        x1, y1, x2, y2 = tuple(map(int, (x1, y1, x2, y2)))

        # Extract the crop
        crop = image[:, y1:y2, x1:x2]
        
        # Resize the crop
        crop = F.resize(crop, self._resize, antialias=True)
        
        # Compute the new bounding box coordinates
        x_bbox_1_new, y_bbox_1_new = x_bbox_1 - x1, y_bbox_1 - y1
        x_bbox_2_new, y_bbox_2_new = x_bbox_2 - x1, y_bbox_2 - y1
        new_bbox = np.array([
            [x_bbox_1_new, y_bbox_1_new],
            [x_bbox_2_new, y_bbox_2_new],
        ])
        # Resizing
        new_bbox = new_bbox * (self._resize[1] / bbox_w)
        
        return crop, new_bbox


if __name__ == "__main__":
    pass
