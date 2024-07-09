# Standard libraries
from time import time

# Third-party libraries
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Custom libraries
import pym3t
from pym3t_ext.toolbox.modules.object_segmentation_prediction_module import (
    ObjectSegmentationPredictionModule,
    BatchInferenceData,
)
from pym3t_ext.toolbox.modules.probabilistic_segmentation_mlp import (
    ProbabilisticSegmentationMLP,
)
from pym3t_ext.toolbox.modules.simple_resnet_module import SimpleResNet
from pym3t_ext.toolbox.utils.partialclass import partialclass
from pym3t_ext.toolbox.utils.crop_resize_transform import CropResizeToAspectTransform


class DeepRegionModality(pym3t.RegionModalityBase):
    """
    Extension of the pym3t.RegionModalityBase class.
    """
    # Model used for prediction
    _prediction_module = None
    
    # Probabilistic segmentation mask
    _predicted_probabilistic_mask = None
    
    # Model trained on 320x240 images
    # width_target = 320
    # height_target = 240
    _resize_transform = CropResizeToAspectTransform(
        resize=(240, 320)
    )
    
    # RBOT images are 640x512
    _original_image_size = (512, 640)
    
    # Set the device to use
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def load_prediction_module(self):
        
        # Load pre-trained weights
        train_module_state_dict = torch.load(
            "weights/probabilistic_segmentation.ckpt"
        ).get("state_dict")
        
        SimpleResNet34 = partialclass(
            SimpleResNet,
            version=34,
            nb_input_channels=4,
            output_logits=True,
        )

        probabilistic_segmentation_model = ProbabilisticSegmentationMLP(
            net_cls=SimpleResNet34,
            patch_size=5,
            mlp_hidden_dims=[128, 64, 32],
            apply_color_transformations=False,
            output_logits=False,
        )
        
        # Instantiate the model used for prediction
        self._prediction_module = ObjectSegmentationPredictionModule(
            probabilistic_segmentation_model=probabilistic_segmentation_model,
        )
        
        def match_state_dict(state_dict: dict, model: torch.nn.Module) -> dict:
            """Extract the state_dict of the model from an other state_dict by matching their
            keys.

            Args:
                state_dict (dict): The state_dict from which to extract the model's state_dict.
                model (torch.nn.Module): The model for which to extract the state_dict.

            Returns:
                dict: The state_dict of the model.
            """
            model_state_dict = model.state_dict()
            new_state_dict = {
                key: value
                for key, value in state_dict.items()
                if key in model_state_dict
            }

            model_state_dict.update(new_state_dict)

            return model_state_dict

        # Get the state_dict of the model used for prediction from the pretrained model
        prediction_module_state_dict = match_state_dict(
            train_module_state_dict,
            self._prediction_module,
        )

        # Load the state_dict into the model
        self._prediction_module.load_state_dict(prediction_module_state_dict)
        
        # Send the model to the device
        self._prediction_module = self._prediction_module.to(self._device)
        
        # Set the model to evaluation mode
        self._prediction_module.eval()
        
    def compute_probabilistic_segmentation(self, bbox: np.ndarray):
        
        # Get RGB image
        cvimage = self.color_camera.image
        
        # Numpy conversion
        image_np = cvimage.to_numpy()
        
        # BGR to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # PyTorch conversion
        image_pytorch = torch.from_numpy(image_np)
        
        # TODO: to change
        # bbox = [155, 100, 200, 150] 
        
        # Set the input data for the prediction module
        input = BatchInferenceData(
            rgbs=image_pytorch.permute(2, 0, 1).unsqueeze(0),
            contour_points_list=[
                    # First example of the batch
                    [bbox,],
                    # Second example of the batch...
                ],
        )
        
        # Resize the input data
        input = self._resize_transform(input)
        
        # Send the input data to the device
        input.rgbs = input.rgbs.to(self._device)
        
        # Perform the prediction
        predicted_probabilistic_masks = self._prediction_module(
            input,
            pixel_segmentation_only=False,
        )
        
        # Convert the first predicted probabilistic mask to numpy
        self._predicted_probabilistic_mask =\
            predicted_probabilistic_masks[0].cpu().numpy()
        
        # # Display the image
        # _, ax = plt.subplots()
        # ax.imshow(input.rgbs.squeeze().permute(1, 2, 0).cpu().numpy())
        # rect = patches.Rectangle(
        #     (bbox[0], bbox[1]),
        #     bbox[2] - bbox[0],
        #     bbox[3] - bbox[1],
        #     linewidth=1,
        #     edgecolor="r",
        #     facecolor="none",
        # )
        # ax.add_patch(rect)
        # ax.axis("off")
        # plt.title("Input image with bounding box")
        # plt.tight_layout()
         
        # Display the mask
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Get the image to set the color scale
        img = ax.imshow(self._predicted_probabilistic_mask, cmap="magma")
        ax.axis("off")

        # Colorbar
        fig.colorbar(img, cmap="magma", cax=cax)

        plt.tight_layout()
        
        plt.show()
    

    def calculate_correspondences(self, iteration: int, corr_iteration: int) -> bool:
        """Calculate the correspondence lines data (center, normal, segment
        probabilities, etc.) for a given iteration. It is an override of the
        pym3t.RegionModalityBase::CalculateCorrespondences method.

        Args:
            iteration (int): Update iteration number.
            corr_iteration (int): Correspondence iteration number.

        Returns:
            bool: Whether the calculation was successful.
        """
        # Load the prediction module if it is not loaded yet
        if self._prediction_module is None:
            self.load_prediction_module() 
        
        # Check if the modality is set up (i.e., required objects are set up and
        # correctly configured, and pre-calculated variables are available
        if not self.IsSetup():
            return False
        
        # Compute body to camera(s) pose(s)
        self.PrecalculatePoseVariables()
        # Set the current scale of the correspondence lines
        # (coarse, medium, fine, etc.)
        self.PrecalculateIterationDependentVariables(corr_iteration)

        # Check if body is visible and fetch images from renderers
        body_visible_depth = False
        if self.model_occlusions:
            body_visible_depth = self.depth_renderer.IsBodyVisible(self.body.name)
            if body_visible_depth:
                self.depth_renderer.FetchDepthImage()

        body_visible_silhouette = False
        if self.use_region_checking:
            body_visible_silhouette =\
                self.silhouette_renderer.IsBodyVisible(self.body.name)
            if body_visible_silhouette:
                self.silhouette_renderer.FetchSilhouetteImage()

        # Search closest template view
        _, view = self.region_model.GetClosestView(self.body2camera_pose)
        data_model_points = view.data_points
        
        # Scale number of lines with contour_length ratio
        n_lines = self.n_lines_max
        if self.use_adaptive_coverage:
            if self.reference_contour_length > 0.0:
                n_lines = int(
                    self.n_lines_max * min(
                        1.0,
                        view.contour_length / self.reference_contour_length
                    ))
            else:
                n_lines = int(
                    self.n_lines_max * view.contour_length\
                        / self.region_model.max_contour_length()
                )
        
        if n_lines > len(data_model_points):
            print("Number of model points too small: "
                  f"{len(data_model_points)} < {n_lines}")
            n_lines = len(data_model_points)
        
        
        #########################################################
        # Compute the probabilistic segmentation only once per image
        if corr_iteration == 0:
            # Find the bounding box of the contour
            bbox = self.ComputeBoundingBox(view.data_points)
            
            # Transform the bounding box coordinates to the probabilistic
            # segmentation image space
            bbox = self._resize_transform.point_transform(
                points=bbox,
                orig_size=self._original_image_size,
            )
            
            # Predict the probabilistic segmentation
            self.compute_probabilistic_segmentation(bbox)
        #########################################################

        # NOTE: will store P(mf|y) and P(mb|y)
        # Initialize segment probabilities
        # segment_probabilities_f = [0] * self.line_length_in_segments
        # segment_probabilities_b = [0] * self.line_length_in_segments

        # Differentiate cases with and without occlusion handling:
        # A. If occlusion handling is enabled:
        #   1. Try to find unoccluded lines. If the number of unoccluded lines is
        #      sufficient, keep them.
        #   2. If the number of unoccluded lines is insufficient, find all lines.
        #
        # B. If occlusion handling is disabled:
        #   1. Find all lines.
        for j in range(2):
            
            self.ClearDataLines()
            
            handle_occlusions = (j == 0) and\
                ((iteration - self.first_iteration) >= self.n_unoccluded_iterations)

            # Iterate over n_lines
            for i in range(n_lines):
                
                # Create a new data line
                data_line = pym3t.DataLine()
                
                # Set line's center, normal, length, etc.
                self.CalculateBasicLineData(data_model_points[i], data_line)
                
                # The line should be long enough, (eventually) not occluded, etc.
                if not self.IsLineValid(
                    data_line,
                    self.use_region_checking and body_visible_silhouette,
                    handle_occlusions and self.model_occlusions,
                    handle_occlusions and body_visible_depth,
                ):
                    continue
                
                # NOTE: we can discard lines that cross the cropped area here (because
                # of the resizing)
                
                #########################################################
                # # Compute the probabilistic segmentations (foreground and background) of
                # # each segment of the line using the color histograms (Bayes' rule)
                # result, segment_probabilities_f, segment_probabilities_b =\
                #     self.CalculateSegmentProbabilities(
                #         segment_probabilities_f,
                #         segment_probabilities_b,
                #         data_line,
                #     )
                # if not result:
                #     continue
                
                # TODO: to put in a custom structure to avoid copying
                # Array to store the line pixels coordinates
                line_pixels_coordinates = np.zeros(
                    (self.line_length, 2),
                    dtype=np.int32,
                )
                
                # Get the line pixels coordinates
                result, line_pixels_coordinates = self.ComputeLinePixelsCoordinates(
                    line_pixels_coordinates,
                    data_line,
                )
                if not result:
                    continue
                
                # Transform the coordinates to the probabilistic segmentation image
                # space
                line_pixels_coordinates = self._resize_transform.point_transform(
                    points=line_pixels_coordinates,
                    orig_size=self._original_image_size,
                )
                
                # print(line_pixels_coordinates.shape)
                # print(self._predicted_probabilistic_mask.shape)
                
                # Compute the probabilistic segmentations (foreground and background) of
                # each pixel of the line using the probabilistic segmentation mask
                line_pixels_probabilities_f = self._predicted_probabilistic_mask[
                    line_pixels_coordinates[:, 1],
                    line_pixels_coordinates[:, 0],
                ]
                line_pixels_probabilities_b = 1 - line_pixels_probabilities_f
                
                # Merge the probabilities of the pixels to get the segment probabilities
                # Nb of pixels per segment : self.scale and I want to multiply the
                # probabilities of the pixels of the same segment
                segment_probabilities_f = np.prod(
                    line_pixels_probabilities_f.reshape(-1, self.scale),
                    axis=1,
                )
                segment_probabilities_b = np.prod(
                    line_pixels_probabilities_b.reshape(-1, self.scale),
                    axis=1,
                )
                
                # Normalize the probabilities
                segment_probabilities_f, segment_probabilities_b =\
                    self.NormalizeSegmentProbabilities(
                        segment_probabilities_f,
                        segment_probabilities_b,
                    )
                #########################################################
                
                # NOTE: exit the for loop, compute the probabilities, and new for
                # loop over the lines to compute the distribution and moments
                
                # Compute the posterior distribution from the probabilistic
                # segmentations and lookup functions (signed distance function
                # + smoothed step function)
                self.CalculateDistribution(
                    segment_probabilities_f,
                    segment_probabilities_b,
                    data_line,
                )
                
                # Compute the mean and variance of the posterior distribution
                # (the noisy posterior distribution will be approximated by a Gaussian
                # distribution during the optimization process)
                self.CalculateDistributionMoments(data_line)
                
                self.AddDataLine(data_line)
                
            if len(self.data_lines) >= self.min_n_unoccluded_lines:
                break
        
        return True
    
    
    def calculate_segment_probabilities_mlp(self):
        pass
