# Third-party libraries
import numpy as np
import pym3t


def main():
    # Set up tracker
    tracker = pym3t.Tracker(
        name="tracker",
        synchronize_cameras=False,
    )

    # Set up renderer geometry
    renderer_geometry = pym3t.RendererGeometry(
        name="renderer geometry",
    )

    # Set up camera
    color_camera = pym3t.LoaderColorCamera(
        name="cam_color",
        metafile_path="config/camera_rbot.yaml",
    )

    # Set up viewers
    color_viewer = pym3t.NormalColorViewer(
        name="color_viewer",
        color_camera=color_camera,
        renderer_geometry=renderer_geometry,
    )
    tracker.AddViewer(color_viewer)
    
    # Set up body
    body = pym3t.Body(
        name="ape",
        geometry_path="data/RBOT/ape/ape.obj",
        geometry_unit_in_meter=0.001,
        geometry_counterclockwise=True,
        geometry_enable_culling=True,
        geometry2body_pose=pym3t.Transform3fA(np.eye(4, dtype=np.float32)),
    )
    renderer_geometry.AddBody(body)

    # Set up region model
    region_model = pym3t.RegionModel(
        name="ape_region_model",
        body=body,
        model_path="tmp/ape_region_model.bin",
    )

    # Set up region modality
    region_modality = pym3t.RegionModality(
        name="ape_region_modality",
        body=body,
        color_camera=color_camera,
        region_model=region_model,
    )

    # Set up link
    link = pym3t.Link(
        name="ape_link",
        body=body,
    )
    link.AddModality(region_modality)
    
    # Set up optimizer
    optimizer = pym3t.Optimizer(
        name="ape_optimizer",
        root_link=link,
    )
    tracker.AddOptimizer(optimizer)
    
    # GT pose read from the dataset
    body2world_pose = pym3t.Transform3fA(
        np.array([
            [0.997056, -0.04307, 0.0634383, 0.019051],
            [0.043157, 0.999068, 0, 0.00408901],
            [-0.0633792, 0.0027378, 0.997986, 0.549],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    )

    # Set up detector
    detector = pym3t.StaticDetector(
        name="detector",
        optimizer=optimizer,
        link2world_pose=body2world_pose,
        reset_joint_poses=False,
    )
    tracker.AddDetector(detector)
    
    # Start tracking
    if not tracker.SetUp():
        return -1

    if not tracker.RunTrackerProcess(True, True): 
        return -1
    
    return 0


if __name__ == "__main__":
    main()
