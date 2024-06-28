#ifndef M3T_INCLUDE_M3T_REGION_MODALITY_BASE_H_
#define M3T_INCLUDE_M3T_REGION_MODALITY_BASE_H_

#include <m3t/modality.h>
#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/common.h>
#include <m3t/region_model.h>
#include <m3t/renderer.h>
#include <m3t/silhouette_renderer.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


namespace m3t {

class RegionModalityBase : public Modality {
    public:
        // Constructors and setup methods
        RegionModalityBase(const std::string &name,
                           const std::shared_ptr<Body> &body_ptr,
                           const std::shared_ptr<ColorCamera> &color_camera_ptr,
                           const std::shared_ptr<RegionModel> &region_model_ptr);
        bool SetUp() override;

        // Setters referenced data
        // void set_color_camera_ptr(
        //     const std::shared_ptr<ColorCamera> &color_camera_ptr);
        // void set_region_model_ptr(
        //     const std::shared_ptr<RegionModel> &region_model_ptr);
        
        // Main methods
        bool StartModality(int iteration, int corr_iteration) override;
        bool CalculateCorrespondences(int iteration, int corr_iteration) override;
        bool VisualizeCorrespondences(int save_idx) override;
        bool CalculateGradientAndHessian(int iteration, int corr_iteration,
                                                 int opt_iteration) override;
        bool VisualizeOptimization(int save_idx) override;
        bool CalculateResults(int iteration) override;
        bool VisualizeResults(int save_idx) override;

        // Getters data
        const std::shared_ptr<ColorCamera> &color_camera_ptr() const;
        const std::shared_ptr<RegionModel> &region_model_ptr() const;
        std::shared_ptr<Model> model_ptr() const override;
        std::vector<std::shared_ptr<Camera>> camera_ptrs() const override;
        std::vector<std::shared_ptr<Renderer>> start_modality_renderer_ptrs()
            const override;
        std::vector<std::shared_ptr<Renderer>> correspondence_renderer_ptrs()
            const override;
        std::vector<std::shared_ptr<Renderer>> results_renderer_ptrs() const override;
        std::shared_ptr<ColorHistograms> color_histograms_ptr() const override;
    
    private:
        // Pointers to referenced objects
        std::shared_ptr<ColorCamera> color_camera_ptr_ = nullptr;
        std::shared_ptr<DepthCamera> depth_camera_ptr_ = nullptr;
        std::shared_ptr<RegionModel> region_model_ptr_ = nullptr;
        std::shared_ptr<FocusedDepthRenderer> depth_renderer_ptr_ = nullptr;
        std::shared_ptr<ColorHistograms> color_histograms_ptr_ = nullptr;
        std::shared_ptr<FocusedSilhouetteRenderer> silhouette_renderer_ptr_ = nullptr;

        // Parameters for histogram calculation
        bool use_shared_color_histograms_ = false;
};
    
}  // namespace m3t

#endif  // M3T_INCLUDE_M3T_REGION_MODALITY_BASE_H_