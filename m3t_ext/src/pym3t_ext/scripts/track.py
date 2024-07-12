# Standard libraries
from pathlib import Path

# Third-party libraries
import numpy as np
import pym3t
import omegaconf

# Custom modules and libraries
from pym3t_ext.components.pytracker import PyTracker
from pym3t_ext.components.deep_region_modality import DeepRegionModality


def track(config: omegaconf.DictConfig) -> int:
    """Track a model in a scene.

    Args:
        config (omegaconf.DictConfig): Configuration parameters.

    Raises:
        ValueError: If the dataset name is not recognized.

    Returns:
        int: Return code (0 if successful, -1 otherwise)
    """
    # Set up tracker
    tracker = PyTracker(
        name="tracker",
        synchronize_cameras=False,
    )

    # Set up renderer geometry
    renderer_geometry = pym3t.RendererGeometry(
        name="renderer_geometry",
    )

    if config.dataset_name == "rbot":
        image_name_pre = config.scene
        n_leading_zeros = 4
        load_directory = Path(config.dataset_dir) / config.model / "frames"
    elif config.dataset_name == "bcot":
        raise NotImplementedError("Evaluation for BCOT dataset is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset_name}")

    # Create intrinsics object
    intrinsics = pym3t.Intrinsics(**config.intrinsics)
    # Set up camera
    color_camera = pym3t.LoaderColorCamera(
        name="color_camera",
        load_directory=load_directory,
        intrinsics=intrinsics,
        image_name_pre=image_name_pre,
        load_index=0,
        n_leading_zeros=n_leading_zeros,
    )

    # Set up viewers
    color_viewer = pym3t.NormalColorViewer(
        name="color_viewer",
        color_camera=color_camera,
        renderer_geometry=renderer_geometry,
    )
    tracker.AddViewer(color_viewer)
    
    if config.dataset_name == "rbot":
        geometry_path =\
            Path(config.dataset_dir) / config.model / f"{config.model}.obj"
    elif config.dataset_name == "bcot":
        raise NotImplementedError("Evaluation for BCOT dataset is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset_name}")
    
    # Set up body
    body = pym3t.Body(
        name=config.model,
        geometry_path=geometry_path,
        geometry_unit_in_meter=config.geometry_unit_in_meter,
        geometry_counterclockwise=True,
        geometry_enable_culling=True,
        geometry2body_pose=pym3t.Transform3fA(np.eye(4, dtype=np.float32)),
    )
    renderer_geometry.AddBody(body)

    # Set up region model
    region_model = pym3t.RegionModel(
        name=f"{config.model}_region_model",
        body=body,
        model_path=f"tmp/{config.model}_region_model.bin",
    )
    
    # Set up deep region modality
    region_modality = DeepRegionModality(
        name=f"{config.model}_deep_region_modality",
        body=body,
        color_camera=color_camera,
        region_model=region_model,
    )
    
    # Set up link
    link = pym3t.Link(
        name=f"{config.model}_link",
        body=body,
    )
    link.AddModality(region_modality)
    
    # Set up optimizer
    optimizer = pym3t.Optimizer(
        name=f"{config.model}_optimizer",
        root_link=link,
    )
    tracker.AddOptimizer(optimizer) 
    
    # Create the body2world pose object
    body2world_pose = pym3t.Transform3fA(
        np.array(config.body2world_pose, dtype=np.float32)
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

    if not tracker.run_tracker_process(
        execute_detection_and_start_tracking=True,
        display_images=True,
        stop_at_each_image=True,
    ):
        return -1
    
    return 0


if __name__ == "__main__":
    # Configuration file path
    config_path = Path("config/track_example_rbot.yaml")

    # Read config file
    config = omegaconf.OmegaConf.load(config_path)
 
    # Run tracking example
    track(config)
