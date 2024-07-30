# Standard libraries
import random

# Third-party libraries
import numpy as np
from torchvision.transforms.functional import crop, resize

# Custom modules
from pym3t_ext.toolbox.modules.object_segmentation_prediction_module import (
    BatchInferenceData,
)


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
        
    def _transform(self, batch: BatchInferenceData) -> BatchInferenceData:
        """Crop and resize the RGB observations to a target aspect ratio.

        Args:
            batch (BatchInferenceData): Batch of data.

        Raises:
            ValueError: If the RGB images are None.

        Returns:
            BatchInferenceData: Transformed batch of data.
        """
        if batch.rgbs is None:
            raise ValueError("The RGB images are None.")
        
        h, w = batch.rgbs.shape[2:4]

        # Skip if the image is already at the target size
        if (h, w) == self._resize:
            return batch

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
            box = (x1, y1, x2, y2)
            box = [int(b) for b in box]
            
            # Crop the RGB images
            rgbs = crop(batch.rgbs, box[1], box[0], box[3] - box[1], box[2] - box[0])
        
        else:
            rgbs = batch.rgbs
        
        w_resize, h_resize = max(self._resize), min(self._resize)
        rgbs = resize(rgbs, (h_resize, w_resize), antialias=True)
        
        # Update the RGB images
        batch.rgbs = rgbs
        
        return batch
    
    def __call__(self, seq: BatchInferenceData) -> BatchInferenceData:
        """Apply or not the transformation to the observation given the
        probability `p`.

        Args:
            seq (SequenceSegmentationData): Sequence observation.

        Returns:
            SequenceSegmentationData: Transformed sequence observation.
        """
        if random.random() <= self._p:
            return self._transform(seq)
        else:
            return seq
    
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
