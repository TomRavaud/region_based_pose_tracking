// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// OpenCV
#include <opencv2/core.hpp>

// Other
#include <sstream>

// Nanobind
// Core
#include <nanobind/nanobind.h>
// Implicit type conversions
#include <nanobind/stl/string.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/trampoline.h>

// M3T
#include <m3t/common.h>
#include <m3t/camera.h>
#include <m3t/renderer_geometry.h>
#include <m3t/body.h>
#include <m3t/link.h>
#include <m3t/region_model.h>
#include <m3t/region_modality.h>
#include <m3t/normal_viewer.h>
#include <m3t/static_detector.h>
#include <m3t/tracker.h>
#include <m3t/loader_camera.h>
#include <m3t/color_histograms.h>

// pym3t
#include "pym3t/region_modality_base.h"


namespace nb = nanobind;
using namespace nb::literals;
using namespace m3t;
using namespace Eigen;


// Define a trampoline for the RegionModalityBase class in order to be able to
// override the CalculateCorrespondences and StartModality methods in Python
struct PyRegionModalityBase : RegionModalityBase {
    NB_TRAMPOLINE(RegionModalityBase, 3);

    bool StartModality(int iteration, int corr_iteration) override {
        NB_OVERRIDE_NAME(
            "start_modality",
            StartModality,
            iteration, corr_iteration);
    }
    bool CalculateCorrespondences(int iteration, int corr_iteration) override {
        NB_OVERRIDE_NAME(
            "calculate_correspondences",
            CalculateCorrespondences,
            iteration, corr_iteration);
    }
    bool CalculateResults(int iteration) override {
        NB_OVERRIDE_NAME(
            "calculate_results",
            CalculateResults,
            iteration);
    }
};


NB_MODULE(_pym3t_mod, m){
    m.doc() = "M3T Python bindings";

    // Tracker: main interface with m3t
    nb::class_<Tracker>(m, "Tracker")
        .def(nb::init<const std::string &, int, int, bool, bool,
                      const std::chrono::milliseconds &, int, int>(),
             "name"_a, "n_corr_iterations"_a=5, "n_update_iterations"_a=2,
             "synchronize_cameras"_a=true,
             "start_tracking_after_detection"_a=false,
             "cycle_duration"_a=std::chrono::milliseconds{33},
             "visualization_time"_a=0, "viewer_time"_a=1)
        .def("SetUp", &Tracker::SetUp, "set_up_all_objects"_a=true)
        .def("RunTrackerProcess", &Tracker::RunTrackerProcess,
             "execute_detection"_a=true, "start_tracking"_a=true,
             "names_detecting"_a=nullptr, "names_starting"_a=nullptr)
        .def("AddViewer", &Tracker::AddViewer)
        .def("AddDetector", &Tracker::AddDetector)
        .def("AddOptimizer", &Tracker::AddOptimizer)
        .def("UpdateCameras", &Tracker::UpdateCameras)
        .def("UpdateSubscribers", &Tracker::UpdateSubscribers)
        .def("CalculateConsistentPoses", &Tracker::CalculateConsistentPoses)
        .def("ExecuteDetectingStep", &Tracker::ExecuteDetectingStep)
        .def("ExecuteStartingStep", &Tracker::ExecuteStartingStep)
        .def("ExecuteTrackingStep", &Tracker::ExecuteTrackingStep)
        .def("UpdatePublishers", &Tracker::UpdatePublishers)
        .def("UpdateViewers", &Tracker::UpdateViewers)
        .def("ExecuteDetection", &Tracker::ExecuteDetection,
             "start_tracking"_a, "names_detecting"_a=nullptr,
             "names_starting"_a=nullptr)
        .def_prop_ro("set_up", &Tracker::set_up)
        .def_prop_ro("name", &Tracker::name)
        .def_prop_rw("n_corr_iterations",
                     &Tracker::n_corr_iterations,
                     &Tracker::set_n_corr_iterations)
        .def_prop_rw("n_update_iterations",
                     &Tracker::n_update_iterations,
                     &Tracker::set_n_update_iterations)
        .def_prop_ro("modalities", &Tracker::modality_ptrs)
        ;
    
    // RendererGeometry
    nb::class_<RendererGeometry>(m, "RendererGeometry")
        .def(nb::init<const std::string &>(), "name"_a)
        .def("AddBody", &RendererGeometry::AddBody)
        ;
    
    // Camera -> not constructible, just to enable automatic downcasting
    // and binding of child classes
    nb::class_<Camera>(m, "Camera");
    // ColorCamera -> not constructible, just to enable automatic downcasting
    // and binding of child classes
    nb::class_<ColorCamera, Camera>(m, "ColorCamera");

    // Intrinsic parameters of a camera
    nb::class_<Intrinsics>(m, "Intrinsics")
        .def(nb::init<float, float, float, float, int, int>(),
             "fu"_a, "fv"_a, "ppu"_a, "ppv"_a, "width"_a, "height"_a)
        ;

    // LoaderColorCamera
    nb::class_<LoaderColorCamera, ColorCamera>(m, "LoaderColorCamera")
        .def(nb::init<const std::string &, const std::filesystem::path &>(),
             "name"_a, "metafile_path"_a)
        .def(nb::init<const std::string &, const std::filesystem::path &,
                      const Intrinsics &, const std::string &, int, int,
                      const std::string &, const std::string &>(),
             "name"_a, "load_directory"_a, "intrinsics"_a,
             "image_name_pre"_a="", "load_index"_a=0, "n_leading_zeros"_a=0,
             "image_name_post"_a="", "load_image_type"_a="png")
        .def_prop_ro("image", &LoaderColorCamera::image)
        ;
    
    // Viewer -> not constructible, just to enable automatic downcasting
    // and binding of child classes
    nb::class_<Viewer>(m, "Viewer");
    // ColorViewer -> not constructible, just to enable automatic downcasting
    // and binding of child classes
    nb::class_<ColorViewer, Viewer>(m, "ColorViewer");

    // NormalColorViewer
    nb::class_<NormalColorViewer, ColorViewer>(m, "NormalColorViewer")
        .def(nb::init<const std::string &, const std::shared_ptr<ColorCamera> &,
                      const std::shared_ptr<RendererGeometry> &, float>(),
             "name"_a, "color_camera"_a, "renderer_geometry"_a, "opacity"_a=0.5f)
        ;
    
    // Bind Transform3fA in order to be able to create a Body object from Python
    nb::class_<Transform3fA>(m, "Transform3fA")
        .def(nb::init<const Matrix4f &>())
        .def("__repr__", [](const Transform3fA &t) {
            std::ostringstream oss;
            oss << t.matrix();
            return "Transform3fA(\n" + oss.str() + "\n)";
        })
        .def_prop_ro("matrix", [](const Transform3fA &t) {
            return t.matrix();
        })
        ;
    
    // Body
    nb::class_<Body>(m, "Body")
      // Constructors and initialization methods
        .def(nb::init<const std::string &, const std::filesystem::path &, float,
                      bool, bool, const Transform3fA &>(),
             "name"_a, "geometry_path"_a, "geometry_unit_in_meter"_a,
             "geometry_counterclockwise"_a, "geometry_enable_culling"_a,
             "geometry2body_pose"_a)
        .def_prop_rw("body2world_pose", &Body::body2world_pose, &Body::set_body2world_pose)
        .def_prop_ro("name", &Body::name)
        ;

    // Link
    nb::class_<Link>(m, "Link")
        .def(nb::init<const std::string &, const std::shared_ptr<Body> &, 
                      const Transform3fA &, const Transform3fA &, const Transform3fA &,
                      const std::array<bool, 6> &, bool>(), 
             "name"_a, "body"_a, "body2joint_pose"_a=Transform3fA::Identity(),
             "joint2parent_pose"_a=Transform3fA::Identity(),
             "link2world_pose"_a=Transform3fA::Identity(),
             "free_directions"_a=std::array<bool, 6>({true, true, true, true, true, true}),
             "fixed_body2joint_pose"_a=true)
        .def("AddModality", &Link::AddModality)
        ;
    
    // View structure
    nb::class_<RegionModel::View>(m, "View")
        .def_ro("data_points", &RegionModel::View::data_points)
        .def_ro("contour_length", &RegionModel::View::contour_length)
        ;
    
    // DataPoint structure
    nb::class_<RegionModel::DataPoint>(m, "DataPoint")
        ;
    
    // RegionModel
    nb::class_<RegionModel>(m, "RegionModel")
        .def(nb::init<const std::string &, const std::shared_ptr<Body> &,
                      const std::filesystem::path &, float, int, int, float, float,
                      bool, int>(),
             "name"_a, "body"_a, "model_path"_a, "sphere_radius"_a=0.8f, 
             "n_divides"_a=4, "n_points_max"_a=200, "max_radius_depth_offset"_a=0.05f,
             "stride_depth_offset"_a=0.002f, "use_random_seed"_a=false,
             "image_size"_a=2000)
        // Use a lambda as a workaround because the pointer to pointer is not supported
        // by nanobind
        .def("GetClosestView",
             [](const RegionModel &self, const Transform3fA &body2camera_pose) {
                    const RegionModel::View *closest_view;
                    // const RegionModel::View* closest_view = nullptr;
                    bool result = self.GetClosestView(body2camera_pose, &closest_view);
                    return std::make_tuple(result, closest_view);
             }, nb::rv_policy::copy)
        .def_prop_ro("max_contour_length", &RegionModel::max_contour_length)
        ;

    // Modality -> not constructible, just to enable automatic downcasting
    // and binding of child classes
    nb::class_<Modality>(m, "Modality");

    // RegionModality
    nb::class_<RegionModality, Modality>(m, "RegionModality")
        .def(nb::init<const std::string &, const std::shared_ptr<Body> &,
                      const std::shared_ptr<ColorCamera> &,
                      const std::shared_ptr<RegionModel> &>(),
             "name"_a, "body"_a, "color_camera"_a, "region_model"_a)
        .def("StartModality", &RegionModality::StartModality)
        .def_prop_rw("visualize_lines_correspondence",
                     &RegionModality::visualize_lines_correspondence,
                     &RegionModality::set_visualize_lines_correspondence)
        .def_prop_rw("visualize_points_optimization",
                     &RegionModality::visualize_points_optimization,
                     &RegionModality::set_visualize_points_optimization)
        .def_prop_rw("function_amplitude",
                     &RegionModality::function_amplitude,
                     &RegionModality::set_function_amplitude)
        .def_prop_rw("function_slope",
                     &RegionModality::function_slope,
                     &RegionModality::set_function_slope)
        .def_prop_rw("scales",
                     &RegionModality::scales,
                     &RegionModality::set_scales)
        .def_prop_rw("standard_deviations",
                     &RegionModality::standard_deviations,
                     &RegionModality::set_standard_deviations)
        ;
    
    // Optimizer
    nb::class_<Optimizer>(m, "Optimizer")
        .def(nb::init<const std::string &, const std::shared_ptr<Link> &,
                      float, float>(),
             "name"_a, "root_link"_a, "tikhonov_parameter_rotation"_a=1000.0f,
             "tikhonov_parameter_translation"_a=30000.0f)
        ;
    
    // Detector -> not constructible, just to enable automatic downcasting
    // and binding of child classes
    nb::class_<Detector>(m, "Detector");

    // StaticDetector
    nb::class_<StaticDetector, Detector>(m, "StaticDetector")
        .def(nb::init<const std::string &, const std::shared_ptr<Optimizer> &,
                      const Transform3fA &, bool>(),
             "name"_a, "optimizer"_a, "link2world_pose"_a, "reset_joint_poses"_a)
        ;
    
    // ColorHistograms
    nb::class_<ColorHistograms>(m, "ColorHistograms")
        .def(nb::init<const std::string &, int, float, float>(),
             "name"_a, "n_bins"_a=16, "learning_rate_f"_a=0.2f, "learning_rate_b"_a=0.2f)
        .def("ClearMemory", &ColorHistograms::ClearMemory)
        .def("InitializeHistograms", &ColorHistograms::InitializeHistograms)
        .def("UpdateHistograms", &ColorHistograms::UpdateHistograms)
        ;


    // DataLine structure
    nb::class_<RegionModalityBase::DataLine>(m, "DataLine")
        .def(nb::init<>())
        .def_rw("center_u", &RegionModalityBase::DataLine::center_u)
        .def_rw("center_v", &RegionModalityBase::DataLine::center_v)
        .def_rw("normal_u", &RegionModalityBase::DataLine::normal_u)
        .def_rw("normal_v", &RegionModalityBase::DataLine::normal_v)
        .def_rw("normal_component_to_scale",
                     &RegionModalityBase::DataLine::normal_component_to_scale)
        .def_rw("delta_r", &RegionModalityBase::DataLine::delta_r)
        .def_rw("distribution", &RegionModalityBase::DataLine::distribution)
        .def_rw("mean", &RegionModalityBase::DataLine::mean)
        .def_rw("measured_variance", &RegionModalityBase::DataLine::measured_variance)
        ;
    
    // RegionModalityBase
    nb::class_<RegionModalityBase, Modality, PyRegionModalityBase>(m, "RegionModalityBase")
        .def(nb::init<const std::string &, const std::shared_ptr<Body> &,
                      const std::shared_ptr<ColorCamera> &,
                      const std::shared_ptr<RegionModel> &>(),
             "name"_a, "body"_a, "color_camera"_a, "region_model"_a)
        .def("IsSetup", &RegionModalityBase::IsSetup)
        .def("PrecalculatePoseVariables", &RegionModalityBase::PrecalculatePoseVariables)
        .def("PrecalculateIterationDependentVariables",
             &RegionModalityBase::PrecalculateIterationDependentVariables)
        .def("CalculateBasicLineData", &RegionModalityBase::CalculateBasicLineData)
        .def("IsLineValid", &RegionModalityBase::IsLineValid)
        // Wrapper lambda function to avoid the mutable reference issue with
        // type casters (segment_probabilities_i is a std::vector<>,
        // data_line.normal_component_to_scale and data_line.delta_r are float).
        // segment_probabilities_i are returned and data_line which is of bound
        // type DataLine is passed instead of its fields (bound types do not suffer
        // from the mutable reference issue)
        .def("CalculateSegmentProbabilities", [](
            const RegionModalityBase &self,
            std::vector<float> *segment_probabilities_f,
            std::vector<float> *segment_probabilities_b,
            RegionModalityBase::DataLine &data_line) {
                bool result = self.CalculateSegmentProbabilities(
                    data_line.center_u, data_line.center_v,
                    data_line.normal_u, data_line.normal_v,
                    segment_probabilities_f, segment_probabilities_b,
                    &data_line.normal_component_to_scale, &data_line.delta_r);
                return std::make_tuple(
                    result,
                    segment_probabilities_f,
                    segment_probabilities_b);
            })
        .def("NormalizeSegmentProbabilities", [](
            const RegionModalityBase &self,
            std::vector<float> *segment_probabilities_f,
            std::vector<float> *segment_probabilities_b) {
                self.NormalizeSegmentProbabilities(
                    segment_probabilities_f,
                    segment_probabilities_b);
                return std::make_tuple(
                    segment_probabilities_f,
                    segment_probabilities_b);
            })
        .def("ComputeBoundingBox", [](
            const RegionModalityBase &self,
            const std::vector<RegionModel::DataPoint> &data_points) {
                Eigen::Matrix<float, 2, 2> bounding_box;
                self.ComputeBoundingBox(data_points, bounding_box);
                return bounding_box;
            })
        .def("ComputeLinePixelsCoordinates", [](
            const RegionModalityBase &self,
            MatrixXi &line_pixels_coordinates,
            RegionModalityBase::DataLine &data_line) {
                bool result = self.ComputeLinePixelsCoordinates(
                    data_line.center_u, data_line.center_v, 
                    data_line.normal_u, data_line.normal_v,
                    line_pixels_coordinates,
                    &data_line.normal_component_to_scale, &data_line.delta_r);
                return std::make_tuple(
                    result,
                    line_pixels_coordinates);
            })
        // Pass bound type (DataLine) to avoid the mutable reference issue
        // with type casters (data_line.distribution is a std::vector<>)
        .def("CalculateDistribution", [](
            const RegionModalityBase &self,
            const std::vector<float> &segment_probabilities_f,
            const std::vector<float> &segment_probabilities_b,
            RegionModalityBase::DataLine &data_line) {
                self.CalculateDistribution(
                    segment_probabilities_f,
                    segment_probabilities_b,
                    &data_line.distribution);
            })
        // Pass bound type (DataLine) to avoid the mutable reference issue
        // with type casters (data_line.mean and data_line.measured_variance are float)
        .def("CalculateDistributionMoments", [](
            const RegionModalityBase &self,
            RegionModalityBase::DataLine &data_line) {
                self.CalculateDistributionMoments(
                    data_line.distribution,
                    &data_line.mean,
                    &data_line.measured_variance);
            })
        .def("AddLinePixelColorsToTempHistograms",
             &RegionModalityBase::AddLinePixelColorsToTempHistograms)
        // New methods to allow data lines manipulation from Python
        .def("AddDataLine", &RegionModalityBase::AddDataLine)
        .def("ClearDataLines", &RegionModalityBase::ClearDataLines)
        .def("UseRegionChecking", &RegionModalityBase::UseRegionChecking)

        .def_prop_ro("body", &RegionModalityBase::body_ptr)
        .def_prop_ro("depth_renderer", &RegionModalityBase::depth_renderer_ptr)
        .def_prop_ro("silhouette_renderer", &RegionModalityBase::silhouette_renderer_ptr)
        .def_prop_ro("region_model", &RegionModalityBase::region_model_ptr)
        .def_prop_ro("color_camera", &RegionModalityBase::color_camera_ptr)
        .def_prop_ro("color_histograms", &RegionModalityBase::color_histograms_ptr_internal)
        .def_prop_ro("model_occlusions", &RegionModalityBase::model_occlusions)
        .def_prop_rw("first_iteration",
                     &RegionModalityBase::first_iteration,
                     &RegionModalityBase::set_first_iteration)
        .def_prop_ro("n_lines_max", &RegionModalityBase::n_lines_max)
        .def_prop_ro("use_adaptive_coverage", &RegionModalityBase::use_adaptive_coverage)
        .def_prop_ro("reference_contour_length", &RegionModalityBase::reference_contour_length)
        .def_prop_ro("use_region_checking", &RegionModalityBase::use_region_checking)
        .def_prop_ro("n_unoccluded_iterations", &RegionModalityBase::n_unoccluded_iterations)
        .def_prop_ro("measure_occlusions", &RegionModalityBase::measure_occlusions)
        .def_prop_ro("min_n_unoccluded_lines", &RegionModalityBase::min_n_unoccluded_lines)
        .def_prop_ro("line_length_in_segments", &RegionModalityBase::line_length_in_segments)
        .def_prop_ro("line_length", &RegionModalityBase::line_length)
        .def_prop_ro("scale", &RegionModalityBase::scale)
        .def_prop_ro("body2camera_pose", &RegionModalityBase::body2camera_pose)
        .def_prop_ro("data_lines", &RegionModalityBase::data_lines)
        .def_prop_ro("use_shared_color_histograms",
                     &RegionModalityBase::use_shared_color_histograms)
        ;
    
    // Renderer -> not constructible, just to enable automatic downcasting
    // and binding of child classes
    nb::class_<Renderer>(m, "Renderer");
    // FocusedRenderer -> not constructible, just to enable automatic downcasting
    // and binding of child classes
    nb::class_<FocusedRenderer, Renderer>(m, "FocusedRenderer")
        .def("AddReferencedBody", &FocusedRenderer::AddReferencedBody)
        ;
    // FocusedDepthRenderer
    nb::class_<FocusedDepthRenderer, FocusedRenderer>(m, "FocusedDepthRenderer")
        .def("IsBodyVisible", &FocusedDepthRenderer::IsBodyVisible)
        .def("FetchDepthImage", &FocusedDepthRenderer::FetchDepthImage)
        ;
    
    // Commonly used enums
    nb::enum_<IDType>(m, "IDType")
        .value("BODY", IDType::BODY)
        .value("REGION", IDType::REGION)
        ;

    // FocusedSilhouetteRenderer
    nb::class_<FocusedSilhouetteRenderer, FocusedDepthRenderer>(m, "FocusedSilhouetteRenderer")
        .def(nb::init<const std::string &, const std::shared_ptr<RendererGeometry> &,
                      const Transform3fA &, const Intrinsics &, IDType, int, float,
                      float>(),
                "name"_a, "renderer_geometry"_a, "world2camera_pose"_a, "intrinsics"_a,
                "id_type"_a=IDType::REGION, "image_size"_a=200, "z_min"_a=0.02f,
                "z_max"_a=10.0f)
        .def("IsBodyVisible", &FocusedSilhouetteRenderer::IsBodyVisible)
        .def("FetchSilhouetteImage", &FocusedSilhouetteRenderer::FetchSilhouetteImage)
        ;

    // Bind cv::Mat in order to be able to convert it to a ndarray
    nb::class_<cv::Mat>(m, "cvMat")
        .def("to_numpy", [](const cv::Mat &mat) {
            // Check the type of mat and convert accordingly
            if (mat.type() == CV_8UC3) {
                size_t shape[3] = {mat.rows, mat.cols, mat.channels()};

                return nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, 3>>(
                    mat.data, 3, shape, nb::capsule());
                
            } else {
                throw std::invalid_argument("Unsupported Mat type");
            }
        })
        ;
}
