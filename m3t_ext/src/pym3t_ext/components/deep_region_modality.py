# Custom libraries
import pym3t


class DeepRegionModality(pym3t.RegionModalityBase):
    """
    Extension of the pym3t.RegionModalityBase class.
    """
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

        # Initialize segment probabilities
        segment_probabilities_f = [0] * self.line_length_in_segments
        segment_probabilities_b = [0] * self.line_length_in_segments

        # Differentiate cases with and without occlusion handling
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
                
                # The line should be long enough, not occluded, etc.
                if not self.IsLineValid(
                    data_line,
                    self.use_region_checking and body_visible_silhouette,
                    handle_occlusions and self.model_occlusions,
                    handle_occlusions and body_visible_depth,
                ):
                    continue
                
                # Compute the probabilistic segmentations (foreground and background) of
                # each segment of the line
                result, segment_probabilities_f, segment_probabilities_b =\
                    self.CalculateSegmentProbabilities(
                        segment_probabilities_f,
                        segment_probabilities_b,
                        data_line,
                    )
                if not result:
                    continue
                
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
