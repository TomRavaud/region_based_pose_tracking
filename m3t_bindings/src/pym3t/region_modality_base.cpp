#include "pym3t/region_modality_base.h"


namespace m3t {

RegionModalityBase::RegionModalityBase(
    const std::string &name, const std::shared_ptr<Body> &body_ptr,
    const std::shared_ptr<ColorCamera> &color_camera_ptr,
    const std::shared_ptr<RegionModel> &region_model_ptr)
    : Modality{name, body_ptr},
      color_camera_ptr_{color_camera_ptr},
      region_model_ptr_{region_model_ptr} {}

bool RegionModalityBase::SetUp() {
//   set_up_ = false;
//   if (!metafile_path_.empty())
//     if (!LoadMetaData()) return false;

//   // Check if all required objects are set up
//   if (!body_ptr_->set_up()) {
//     std::cerr << "Body " << body_ptr_->name() << " was not set up" << std::endl;
//     return false;
//   }
//   if (!region_model_ptr_->set_up()) {
//     std::cerr << "Region model " << region_model_ptr_->name()
//               << " was not set up" << std::endl;
//     return false;
//   }
//   if (!color_camera_ptr_->set_up()) {
//     std::cerr << "Color camera " << color_camera_ptr_->name()
//               << " was not set up" << std::endl;
//     return false;
//   }
//   if (measure_occlusions_ && !depth_camera_ptr_->set_up()) {
//     std::cerr << "Depth camera " << depth_camera_ptr_->name()
//               << " was not set up" << std::endl;
//     return false;
//   }
//   if (model_occlusions_ && !depth_renderer_ptr_->set_up()) {
//     std::cerr << "Focused depth renderer " << depth_renderer_ptr_->name()
//               << " was not set up" << std::endl;
//     return false;
//   }
//   if (use_shared_color_histograms_ && !color_histograms_ptr_->set_up()) {
//     std::cerr << "Color histogram " << color_histograms_ptr_->name()
//               << " was not set up" << std::endl;
//     return false;
//   }
//   if (use_region_checking_ && !silhouette_renderer_ptr_->set_up()) {
//     std::cerr << "Focused silhouette renderer "
//               << silhouette_renderer_ptr_->name() << " was not set up"
//               << std::endl;
//     return false;
//   }

//   // Check if all required objects are correctly configured
//   if (model_occlusions_ &&
//       !depth_renderer_ptr_->IsBodyReferenced(body_ptr_->name())) {
//     std::cerr << "Focused depth renderer " << depth_renderer_ptr_->name()
//               << " does not reference body " << body_ptr_->name() << std::endl;
//     return false;
//   }
//   if (use_region_checking_ &&
//       silhouette_renderer_ptr_->id_type() != IDType::REGION) {
//     std::cerr << "Focused silhouette renderer "
//               << silhouette_renderer_ptr_->name()
//               << " does not use id_type REGION" << std::endl;
//   }
//   if (use_region_checking_ &&
//       !silhouette_renderer_ptr_->IsBodyReferenced(body_ptr_->name())) {
//     std::cerr << "Focused silhouette renderer "
//               << silhouette_renderer_ptr_->name() << " does not reference body "
//               << body_ptr_->name() << std::endl;
//     return false;
//   }

//   PrecalculateFunctionLookup();
//   PrecalculateDistributionVariables();
//   if (!use_shared_color_histograms_)
//     if (!SetUpInternalColorHistograms()) return false;
//   SetImshowVariables();
//   PrecalculateCameraVariables();
//   if (!PrecalculateModelVariables()) return false;
//   PrecalculateRendererVariables();

//   set_up_ = true;
  return true;
}

bool RegionModalityBase::StartModality(int iteration, int corr_iteration) {
//   if (!IsSetup()) return false;

//   first_iteration_ = iteration;
//   PrecalculatePoseVariables();

//   // Initialize histograms
//   bool handle_occlusions = n_unoccluded_iterations_ == 0;
//   if (!use_shared_color_histograms_) color_histograms_ptr_->ClearMemory();
//   AddLinePixelColorsToTempHistograms(handle_occlusions);
//   if (!use_shared_color_histograms_)
//     color_histograms_ptr_->InitializeHistograms();
  return true;
}

bool RegionModalityBase::CalculateCorrespondences(int iteration,
                                                  int corr_iteration) {
//   if (!IsSetup()) return false;

//   PrecalculatePoseVariables();
//   PrecalculateIterationDependentVariables(corr_iteration);

//   // Check if body is visible and fetch images from renderers
//   bool body_visible_depth;
//   if (model_occlusions_) {
//     body_visible_depth = depth_renderer_ptr_->IsBodyVisible(body_ptr_->name());
//     if (body_visible_depth) depth_renderer_ptr_->FetchDepthImage();
//   }
//   bool body_visible_silhouette;
//   if (use_region_checking_) {
//     body_visible_silhouette =
//         silhouette_renderer_ptr_->IsBodyVisible(body_ptr_->name());
//     if (body_visible_silhouette)
//       silhouette_renderer_ptr_->FetchSilhouetteImage();
//   }

//   // Search closest template view
//   const RegionModel::View *view;
//   region_model_ptr_->GetClosestView(body2camera_pose_, &view);
//   auto &data_model_points{view->data_points};

//   // Scale number of lines with contour_length ratio
//   int n_lines = n_lines_max_;
//   if (use_adaptive_coverage_) {
//     if (reference_contour_length_ > 0.0f)
//       n_lines = n_lines_max_ * std::min(1.0f, view->contour_length /
//                                                   reference_contour_length_);
//     else
//       n_lines = n_lines_max_ * view->contour_length /
//                 region_model_ptr_->max_contour_length();
//   }
//   if (n_lines > data_model_points.size()) {
//     std::cerr << "Number of model points too small: "
//               << data_model_points.size() << " < " << n_lines << std::endl;
//     n_lines = data_model_points.size();
//   }

//   // Differentiate cases with and without occlusion handling
//   std::vector<float> segment_probabilities_f(line_length_in_segments_);
//   std::vector<float> segment_probabilities_b(line_length_in_segments_);
//   for (int j = 0; j < 2; ++j) {
//     data_lines_.clear();
//     bool handle_occlusions =
//         j == 0 && (iteration - first_iteration_) >= n_unoccluded_iterations_;

//     // Iterate over n_lines
//     for (int i = 0; i < n_lines; ++i) {
//       DataLine data_line;
//       CalculateBasicLineData(data_model_points[i], &data_line);
//       if (!IsLineValid(
//               data_line, use_region_checking_ && body_visible_silhouette,
//               handle_occlusions && measure_occlusions_,
//               handle_occlusions && model_occlusions_ && body_visible_depth))
//         continue;
//       if (!CalculateSegmentProbabilities(
//               data_line.center_u, data_line.center_v, data_line.normal_u,
//               data_line.normal_v, &segment_probabilities_f,
//               &segment_probabilities_b, &data_line.normal_component_to_scale,
//               &data_line.delta_r))
//         continue;
//       CalculateDistribution(segment_probabilities_f, segment_probabilities_b,
//                             &data_line.distribution);

//       CalculateDistributionMoments(data_line.distribution, &data_line.mean,
//                                    &data_line.measured_variance);
//       data_lines_.push_back(std::move(data_line));
//     }
//     if (data_lines_.size() >= min_n_unoccluded_lines_) break;
//   }
  return true;
}

bool RegionModalityBase::VisualizeCorrespondences(int save_idx) {
//   if (!IsSetup()) return false;

//   if (visualize_lines_correspondence_)
//     VisualizeLines("lines_correspondence", save_idx);
//   if (visualize_points_correspondence_)
//     VisualizePointsColorImage("color_image_correspondence", save_idx);
//   if (visualize_points_depth_image_correspondence_ && measure_occlusions_)
//     VisualizePointsDepthImage("depth_image_correspondence", save_idx);
//   if (visualize_points_depth_rendering_correspondence_ && model_occlusions_)
//     VisualizePointsDepthRendering("depth_rendering_correspondence", save_idx);
//   if (visualize_points_silhouette_rendering_correspondence_ &&
//       use_region_checking_)
//     VisualizePointsSilhouetteRendering("silhouette_rendering_correspondence",
//                                        save_idx);
  return true;
}

bool RegionModalityBase::CalculateGradientAndHessian(int iteration,
                                                     int corr_iteration,
                                                     int opt_iteration) {
//   if (!IsSetup()) return false;

//   PrecalculatePoseVariables();
//   gradient_.setZero();
//   hessian_.setZero();

//   // Iterate over correspondence lines
//   for (auto &data_line : data_lines_) {
//     // Calculate point coordinates in camera frame
//     data_line.center_f_camera = body2camera_pose_ * data_line.center_f_body;
//     float x = data_line.center_f_camera(0);
//     float y = data_line.center_f_camera(1);
//     float z = data_line.center_f_camera(2);

//     // Calculate delta_cs
//     float fu_z = fu_ / z;
//     float fv_z = fv_ / z;
//     float xfu_z = x * fu_z;
//     float yfv_z = y * fv_z;
//     float delta_cs = (data_line.normal_u * (xfu_z + ppu_ - data_line.center_u) +
//                       data_line.normal_v * (yfv_z + ppv_ - data_line.center_v) -
//                       data_line.delta_r) *
//                      data_line.normal_component_to_scale;

//     // Calculate first derivative of loglikelihood with respect to delta_cs
//     float dloglikelihood_ddelta_cs;
//     if (opt_iteration < n_global_iterations_) {
//       dloglikelihood_ddelta_cs =
//           (data_line.mean - delta_cs) / data_line.measured_variance;
//     } else {
//       // Calculate distribution indexes
//       // Note: (distribution_length - 1) / 2 + 1 = (distribution_length + 1) / 2
//       int dist_idx_upper = int(delta_cs + distribution_length_plus_1_half_);
//       int dist_idx_lower = dist_idx_upper - 1;
//       if (dist_idx_upper <= 0 || dist_idx_upper >= distribution_length_)
//         continue;

//       dloglikelihood_ddelta_cs =
//           (std::log(data_line.distribution[dist_idx_upper]) -
//            std::log(data_line.distribution[dist_idx_lower])) *
//           learning_rate_ / data_line.measured_variance;
//     }

//     // Calculate first order derivative of delta_cs with respect to theta
//     Eigen::RowVector3f ddelta_cs_dcenter{
//         data_line.normal_component_to_scale * data_line.normal_u * fu_z,
//         data_line.normal_component_to_scale * data_line.normal_v * fv_z,
//         data_line.normal_component_to_scale *
//             (-data_line.normal_u * xfu_z - data_line.normal_v * yfv_z) / z};
//     Eigen::RowVector3f ddelta_cs_dtranslation{ddelta_cs_dcenter *
//                                               body2camera_rotation_};
//     Eigen::Matrix<float, 1, 6> ddelta_cs_dtheta;
//     ddelta_cs_dtheta << data_line.center_f_body.transpose().cross(
//         ddelta_cs_dtranslation),
//         ddelta_cs_dtranslation;

//     // Calculate weight
//     float weight = min_expected_variance_ /
//                    (data_line.normal_component_to_scale *
//                     data_line.normal_component_to_scale * variance_);

//     // Calculate gradient and hessian
//     gradient_ +=
//         (weight * dloglikelihood_ddelta_cs) * ddelta_cs_dtheta.transpose();
//     hessian_.triangularView<Eigen::Lower>() -=
//         (weight / data_line.measured_variance) * ddelta_cs_dtheta.transpose() *
//         ddelta_cs_dtheta;
//   }
//   hessian_ = hessian_.selfadjointView<Eigen::Lower>();
  return true;
}

bool RegionModalityBase::VisualizeOptimization(int save_idx) {
//   if (!IsSetup()) return false;

//   if (visualize_points_optimization_)
//     VisualizePointsColorImage("color_image_optimization", save_idx);
//   if (visualize_points_histogram_image_optimization_)
//     VisualizePointsHistogramImage("histogram_image_optimization", save_idx);
//   if (visualize_gradient_optimization_) VisualizeGradient();
//   if (visualize_hessian_optimization_) VisualizeHessian();
  return true;
}

bool RegionModalityBase::CalculateResults(int iteration) {
//   if (!IsSetup()) return false;

//   // Calculate histograms
//   if (!use_shared_color_histograms_) color_histograms_ptr_->ClearMemory();
//   PrecalculatePoseVariables();
//   bool handle_occlusions =
//       (iteration - first_iteration_) >= n_unoccluded_iterations_;
//   AddLinePixelColorsToTempHistograms(handle_occlusions);
//   if (!use_shared_color_histograms_) color_histograms_ptr_->UpdateHistograms();
  return true;
}

bool RegionModalityBase::VisualizeResults(int save_idx) {
//   if (!IsSetup()) return false;

//   if (visualize_points_result_)
//     VisualizePointsColorImage("color_image_result", save_idx);
//   if (visualize_points_histogram_image_result_)
//     VisualizePointsHistogramImage("histogram_image_result", save_idx);
//   if (visualize_pose_result_) VisualizePose();
  return true;
}

const std::shared_ptr<ColorCamera> &RegionModalityBase::color_camera_ptr() const {
  return color_camera_ptr_;
}

const std::shared_ptr<RegionModel> &RegionModalityBase::region_model_ptr() const {
  return region_model_ptr_;
}

std::shared_ptr<Model> RegionModalityBase::model_ptr() const {
  return region_model_ptr_;
}

std::vector<std::shared_ptr<Camera>> RegionModalityBase::camera_ptrs() const {
  return {color_camera_ptr_, depth_camera_ptr_};
}

std::vector<std::shared_ptr<Renderer>>
RegionModalityBase::start_modality_renderer_ptrs() const {
  return {depth_renderer_ptr_, silhouette_renderer_ptr_};
}

std::vector<std::shared_ptr<Renderer>>
RegionModalityBase::correspondence_renderer_ptrs() const {
  return {depth_renderer_ptr_, silhouette_renderer_ptr_};
}

std::vector<std::shared_ptr<Renderer>> RegionModalityBase::results_renderer_ptrs()
    const {
  return {depth_renderer_ptr_, silhouette_renderer_ptr_};
}

std::shared_ptr<ColorHistograms> RegionModalityBase::color_histograms_ptr() const {
  if (use_shared_color_histograms_)
    return color_histograms_ptr_;
  else
    return nullptr;
}
}  // namespace m3t