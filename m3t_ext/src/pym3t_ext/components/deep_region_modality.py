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
    _segmentation_size = (240, 320)
    _crop_resize_transform = CropResizeToAspectTransform(
        resize=_segmentation_size,
    )
    
    # RBOT images are 640x512
    _original_image_size = (512, 640)
    
    # Set the device to use
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def start_modality(self, iteration: int, corr_iteration: int) -> bool:
        """Start the modality. It is an override of the
        pym3t.RegionModalityBase::StartModality method.

        Args:
            iteration (int): Update iteration number.
            corr_iteration (int): Correspondence iteration number.

        Returns:
            bool: Whether the modality was started successfully.
        """
        if not self.IsSetup():
            return False
        
        self.first_iteration = iteration
        
        # Compute body to camera(s) pose(s)
        self.PrecalculatePoseVariables()
        
        # Initialize histograms (not used)
        handle_occlusions = self.n_unoccluded_iterations == 0
        if not self.use_shared_color_histograms:
            self.color_histograms.ClearMemory()
        self.AddLinePixelColorsToTempHistograms(handle_occlusions)
        if not self.use_shared_color_histograms:
            self.color_histograms.InitializeHistograms()
        
        # Load the prediction module
        if self._prediction_module is None:
            self.load_prediction_module()
        
        #-------------------------------------------------#
        # Initialize the probabilistic segmentation model #
        #-------------------------------------------------#
        # Search closest template view
        _, view = self.region_model.GetClosestView(self.body2camera_pose)
        
        # Find the bounding box of the contour
        bbox = self.ComputeBoundingBox(view.data_points)
        
        # Predict the probabilistic segmentation model
        self.compute_probabilistic_segmentation(
            bbox,
            pixel_segmentation_only=False,
            mlp_parameters_prediction_only=True,
            visualize=True,
        )
        
        return True

    def load_prediction_module(self) -> None:
        """
        Load the prediction module used for probabilistic segmentation.
        """
        # Load pre-trained weights
        train_module_state_dict = torch.load(
            "weights/probabilistic_segmentation_2.ckpt"
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
            mlp_hidden_dims=[64, 32, 16],
            apply_color_transformations=False,
            output_logits=False,
        )
        
        # Instantiate the model used for prediction
        self._prediction_module = ObjectSegmentationPredictionModule(
            probabilistic_segmentation_model=probabilistic_segmentation_model,
        )
        
        def match_state_dict(state_dict: dict, model: torch.nn.Module) -> dict:
            """Extract the state_dict of the model from an other state_dict by matching
            their keys.

            Args:
                state_dict (dict): The state_dict from which to extract the model's
                    state_dict.
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

    def compute_probabilistic_segmentation(
        self,
        bbox: np.ndarray = None,
        pixel_segmentation_only: bool = False,
        mlp_parameters_prediction_only: bool = False,
        visualize: bool = False,
    ) -> None:
        """Compute the probabilistic segmentation of the image, or the MLP parameters
        only.

        Args:
            bbox (np.ndarray): The bounding box of the contour (only needed to predict
                the MLP parameters). Defaults to None.
            pixel_segmentation_only (bool, optional): Whether to recompute the MLP
                parameters or to use the previous ones. Defaults to False.
            mlp_parameters_prediction_only (bool, optional): Whether to predict the
                MLP parameters only, without outputting the pixel segmentation.
                Defaults to False.
            visualize (bool, optional): Whether to visualize the results. Defaults to
                False.

        Raises:
            ValueError: If pixel_segmentation_only and mlp_parameters_prediction_only
                are True at the same time.
        """
        # Get RGB image
        cvimage = self.color_camera.image
        
        # Numpy conversion
        image_np = cvimage.to_numpy()
        
        # BGR to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # PyTorch conversion
        image_pytorch = torch.from_numpy(image_np).permute(2, 0, 1)
        
        if bbox is not None:
            # Crop and resize the image and the bounding box
            image_pytorch, bbox = self._crop_resize_transform(image_pytorch, bbox)
        
        
        # Set the input data for the prediction module
        input = BatchInferenceData(
            rgbs=image_pytorch.unsqueeze(0),
            contour_points_list=[
                    # First example of the batch
                    [bbox,],
                    # Second example of the batch...
                ],
        )
        
        # Send the input data to the device
        input.rgbs = input.rgbs.to(self._device)
        
        # Perform the prediction
        if pixel_segmentation_only and mlp_parameters_prediction_only:
            raise ValueError(
                "pixel_segmentation_only and mlp_parameters_prediction_only "
                "cannot be True at the same time."
            )
        
        elif pixel_segmentation_only:
            predicted_probabilistic_masks = self._prediction_module(
                input,
                pixel_segmentation_only=True,
                mlp_parameters_prediction_only=False,
            )
            # Convert the first predicted probabilistic mask to numpy
            predicted_probabilistic_mask =\
                predicted_probabilistic_masks[0].cpu().numpy()
            
            if visualize:
                # Display the mask
                fig, ax = plt.subplots()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                # Get the image to set the color scale
                img = ax.imshow(predicted_probabilistic_mask, cmap="bwr")
                ax.axis("off")

                # Colorbar
                fig.colorbar(img, cmap="bwr", cax=cax)

                plt.tight_layout()
                plt.show()
            
            return predicted_probabilistic_mask
        
        elif mlp_parameters_prediction_only:
            self._prediction_module(
                input,
                pixel_segmentation_only=False,
                mlp_parameters_prediction_only=True,
            )
            
            if visualize:
                # Display the image
                _, ax = plt.subplots()
                ax.imshow(input.rgbs.squeeze().permute(1, 2, 0).cpu().numpy())
                rect = patches.Rectangle(
                    (bbox[0, 0], bbox[0, 1]),
                    bbox[1, 0] - bbox[0, 0],
                    bbox[1, 1] - bbox[0, 1],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.axis("off")
                plt.tight_layout()
                plt.show()
                
            return
        
        else:
            predicted_probabilistic_masks = self._prediction_module(
                input,
                pixel_segmentation_only=False,
                mlp_parameters_prediction_only=False,
            )
            # Convert the first predicted probabilistic mask to numpy
            predicted_probabilistic_mask =\
                predicted_probabilistic_masks[0].cpu().numpy()
            
            return predicted_probabilistic_mask
        

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
        
        
        # Compute the probabilistic segmentation only once per image
        if corr_iteration == 0:
            
            # Predict the probabilistic segmentation mask using the
            # already computed MLP parameters
            self._predicted_probabilistic_mask =\
                self.compute_probabilistic_segmentation(
                    bbox=None,
                    pixel_segmentation_only=True,
                    mlp_parameters_prediction_only=False,
                    visualize=True
                )
        
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
                
                # Compute the probabilistic segmentations (P(mf|y) and P(mb|y)) of
                # each pixel of the line using the probabilistic segmentation mask
                line_pixels_probabilities_f = self._predicted_probabilistic_mask[
                    line_pixels_coordinates[:, 1],
                    line_pixels_coordinates[:, 0],
                ]
                line_pixels_probabilities_b = 1 - line_pixels_probabilities_f
                
                # Merge the probabilities of the pixels belonging to the same segment to
                # get the segment probabilities
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

    def calculate_results(self, iteration: int) -> bool:
        """Computations to be done at the end of the tracking process for
        the current image (e.g., update of the color histograms, etc.).
        It is an override of the pym3t.RegionModalityBase::CalculateResults method.

        Args:
            iteration (int): Update iteration number.

        Returns:
            bool: Whether the calculation was successful.
        """
        if not self.IsSetup():
            return False
        
        if not self.use_shared_color_histograms:
            self.color_histograms.ClearMemory()
        
        # Compute body to camera(s) pose(s)
        self.PrecalculatePoseVariables()
        
        # Update the color histograms (not used)
        handle_occlusions = (iteration - self.first_iteration)\
            >= self.n_unoccluded_iterations
        self.AddLinePixelColorsToTempHistograms(handle_occlusions)
        if not self.use_shared_color_histograms:
            self.color_histograms.UpdateHistograms()
        
        #---------------------------------------------#
        # Update the probabilistic segmentation model #
        #---------------------------------------------#
        # Search closest template view
        _, view = self.region_model.GetClosestView(self.body2camera_pose)
        
        # Find the bounding box of the contour
        bbox = self.ComputeBoundingBox(view.data_points)
        
        # Predict the probabilistic segmentation model
        self.compute_probabilistic_segmentation(
            bbox,
            pixel_segmentation_only=False,
            mlp_parameters_prediction_only=True,
            visualize=True
        )

        return True
