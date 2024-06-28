// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

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
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>

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

// pym3t
#include "pym3t/region_modality_base.h"


namespace nb = nanobind;
using namespace nb::literals;
using namespace m3t;
using namespace Eigen;


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
    
    // RegionModel
    nb::class_<RegionModel>(m, "RegionModel")
        .def(nb::init<const std::string &, const std::shared_ptr<Body> &,
                      const std::filesystem::path &, float, int, int, float, float,
                      bool, int>(),
             "name"_a, "body"_a, "model_path"_a, "sphere_radius"_a=0.8f, 
             "n_divides"_a=4, "n_points_max"_a=200, "max_radius_depth_offset"_a=0.05f,
             "stride_depth_offset"_a=0.002f, "use_random_seed"_a=false,
             "image_size"_a=2000)
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
        .def_prop_rw("visualize_lines_correspondence",
                     &RegionModality::visualize_lines_correspondence,
                     &RegionModality::set_visualize_lines_correspondence)
        .def_prop_rw("visualize_points_optimization",
                     &RegionModality::visualize_points_optimization,
                     &RegionModality::set_visualize_points_optimization)
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
    
    // RegionModalityBase
    nb::class_<RegionModalityBase, Modality>(m, "RegionModalityBase")
        .def(nb::init<const std::string &, const std::shared_ptr<Body> &,
                      const std::shared_ptr<ColorCamera> &,
                      const std::shared_ptr<RegionModel> &>(),
             "name"_a, "body"_a, "color_camera"_a, "region_model"_a)
        ;
}