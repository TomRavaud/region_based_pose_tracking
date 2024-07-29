# Standard libraries
from dataclasses import dataclass
from typing import Optional
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third party libraries
import torch
import torch.nn as nn
import numpy as np

# Custom modules
from pym3t_ext.toolbox.modules.mobile_sam_module import MobileSAM


@dataclass
class BatchCLinesInferenceData:
    """
    Dataclass for the input data of the ObjectSegmentationPredictionModel.
    """
    # Batch of RGB images
    # Shape: (B, C, H, W)
    # Range: [0, 255]
    # Data type: torch.uint8
    rgbs: torch.Tensor
    
    # One element per image in the batch. An element is a list of contours,
    # where each contour is a numpy array of shape (N, 2)
    contour_points_list: list
    
    # Batch of correspondence lines
    # Shape: (B, C, N, L) where N is the number of lines per image,
    # and L the length of the lines
    # Range: [0, 255]
    # Data type: torch.uint8
    clines_rgbs: torch.Tensor


class ObjectSegmentationCLinesPredictionModel(nn.Module):
    """
    Module that predicts object segmentations using the MobileSAM and ResNet18
    pre-trained models.
    """
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
        compile: bool = False,
    ) -> None:
        """Constructor of the ObjectSegmentationPredictionModel.

        Args:
            probabilistic_segmentation_model (nn.Module): Model that predicts
                probabilistic segmentations.
            compile (bool, optional): Whether to compile parts of the model.
                Defaults to False.
        """
        super().__init__()
        
        # Instantiate the MobileSAM module
        # (for explicit object segmentation alignment)
        self._mobile_sam = MobileSAM(
            sam_checkpoint="weights/mobile_sam.ckpt",
            compile=compile,
        )
        self._binary_masks = None
        
        self._probabilistic_segmentation_model = probabilistic_segmentation_model
    
    @property
    def binary_masks(self) -> Optional[torch.Tensor]:
        """Getter for the binary masks from the MobileSAM model.

        Returns:
            torch.Tensor: Binary masks from the MobileSAM model.
        """
        return self._binary_masks
    
    @torch.no_grad()
    def forward(
        self,
        x: BatchCLinesInferenceData,
        clines_segmentation_only: bool = False,
        context_vectors_prediction_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (BatchCLinesInferenceData): Input data for the model.
            clines_segmentation_only (bool, optional): Whether to use the previous
                segmentation model to predict the correspondence lines segmentation.
                Defaults to False.
            context_vectors_prediction_only (bool, optional): Whether to only predict
                the context vectors. Defaults to False.

        Returns:
            torch.Tensor: Predicted segmentation masks.
        """
        if clines_segmentation_only and context_vectors_prediction_only:
            raise ValueError(
                "Only one of the flags 'clines_segmentation_only' and"
                "'context_vectors_prediction_only' can be True."
            )
        
        elif clines_segmentation_only:
            
            self._binary_masks = None
            
            clines_probabilistic_masks =\
                self._probabilistic_segmentation_model.forward_clines_segmentation(
                    x.clines_rgbs,
                )
            return clines_probabilistic_masks

        elif context_vectors_prediction_only:
            
            # Predict masks, scores and logits using the MobileSAM model
            mobile_sam_outputs = self._mobile_sam(x.rgbs, x.contour_points_list)

            # Stack the masks from the MobileSAM outputs
            self._binary_masks = torch.stack([
                output["masks"][:, torch.argmax(output["iou_predictions"])]
                for output in mobile_sam_outputs
            ])
            
            # Predict the MLP parameters
            self._probabilistic_segmentation_model.forward_context_vectors_prediction(
                x.rgbs,
                self._binary_masks,
            )
            return
        
        else:
            # Predict masks, scores and logits using the MobileSAM model
            mobile_sam_outputs = self._mobile_sam(x.rgbs, x.contour_points_list)

            # Stack the masks from the MobileSAM outputs
            self._binary_masks = torch.stack([
                output["masks"][:, torch.argmax(output["iou_predictions"])]
                for output in mobile_sam_outputs
            ])
            
            # Compute the probabilistic segmentation masks
            probabilistic_masks = self._probabilistic_segmentation_model(
                x.rgbs,
                self._binary_masks,
                x.clines_rgbs,
                
            )
            return probabilistic_masks
    

class ObjectSegmentationCLinesPredictionModule(nn.Module):
    """
    Module that predicts object segmentations using the
    ObjectSegmentationPredictionModel.
    """
    def __init__(
        self,
        probabilistic_segmentation_model: nn.Module,
        compile: bool = False,
    ) -> None:
        """
        Constructor of the ObjectSegmentationPredictionModule.
        
        Args:
            probabilistic_segmentation_model (nn.Module): Model that predicts
                probabilistic segmentations.
            compile (bool, optional): Whether to compile parts of the model.
                Defaults to False.
        """
        super().__init__()
        
        self._model = ObjectSegmentationCLinesPredictionModel(
            probabilistic_segmentation_model=probabilistic_segmentation_model,
            compile=compile,
        )
    
    @property
    def model(self) -> ObjectSegmentationCLinesPredictionModel:
        """Getter for the ObjectSegmentationPredictionModel.

        Returns:
            ObjectSegmentationPredictionModel: ObjectSegmentationPredictionModel.
        """
        return self._model
    
    @torch.no_grad()
    def forward(
        self,
        x: BatchCLinesInferenceData,
        clines_segmentation_only: bool = False,
        context_vectors_prediction_only: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (BatchCLinesInferenceData): Input data for the model.
            clines_segmentation_only (bool, optional): Whether to use the previous
                segmentation model to predict the correspondence lines segmentation.
                Defaults to False.
            context_vectors_prediction_only (bool, optional): Whether to only predict
                the context vectors. Defaults to False.

        Returns:
            torch.Tensor: Predicted segmentation masks.
        """
        return self._model(
            x,
            clines_segmentation_only,
            context_vectors_prediction_only,
        )


if __name__ == "__main__":
    pass
