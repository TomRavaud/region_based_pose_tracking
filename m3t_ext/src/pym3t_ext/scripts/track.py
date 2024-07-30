# Standard libraries
from pathlib import Path
import datetime

# Third-party libraries
import numpy as np
import pym3t
import omegaconf

# Custom modules and libraries
from pym3t_ext.components.pytracker import PyTracker
from pym3t_ext.components.deep_region_modality import DeepRegionModality
from pym3t_ext.components.deep_clines_region_modality import DeepCLinesRegionModality


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
    tracker.n_corr_iterations = 7

    # Set up renderer geometry
    renderer_geometry = pym3t.RendererGeometry(
        name="renderer_geometry",
    )

    if config.dataset_name == "rbot":
        image_name_pre = config.scene
        n_leading_zeros = 4
        load_directory = Path(config.dataset_dir) / config.model / "frames"
        
        # Load camera intrinsics
        camera_intrinsics_file = Path(config.dataset_dir) / "camera_calibration.txt"
        with open(camera_intrinsics_file, "r") as f:
            f.readline()  # Skip the first line
            line = f.readline()
        # Parse intrinsics
        fu, fv, ppu, ppv = line.split("\t")[:4]
        # Create intrinsics dictionary
        intrinsics = {
            "fu": float(fu),
            "fv": float(fv),
            "ppu": float(ppu),
            "ppv": float(ppv),
            "width": config.image_size.width,
            "height": config.image_size.height,
        } 
    elif config.dataset_name == "bcot":
        raise NotImplementedError("Evaluation for BCOT dataset is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset_name}")

    # Create intrinsics object
    intrinsics = pym3t.Intrinsics(**intrinsics)
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
    
    # Set up the modality
    if config.modality == "region_modality":
        region_modality = pym3t.RegionModality(
            name=f"{config.model}_region_modality",
            body=body,
            color_camera=color_camera,
            region_model=region_model,
        )
    elif config.modality == "deep_region_modality":
        region_modality = DeepRegionModality(
            name=f"{config.model}_deep_region_modality",
            body=body,
            color_camera=color_camera,
            region_model=region_model,
        )
    elif config.modality == "deep_clines_region_modality":
        region_modality = DeepCLinesRegionModality(
            name=f"{config.model}_deep_clines_region_modality",
            body=body,
            color_camera=color_camera,
            region_model=region_model,
        )        
    else:
        raise ValueError(f"Unknown modality: {config.modality}")
    
    # TODO: parameter in config file
    use_region_checking = True
    if use_region_checking:
        silhouette_renderer = pym3t.FocusedSilhouetteRenderer(
            name=f"{config.model}_focused_silhouette_renderer",
            renderer_geometry=renderer_geometry,
            camera=color_camera,
        )
        silhouette_renderer.AddReferencedBody(body)
        region_modality.UseRegionChecking(silhouette_renderer)

    # RBOT specific parameters (see Stoiber PhD thesis)
    region_modality.function_amplitude = 0.36
    region_modality.function_slope = 0.0
    region_modality.scales = [5, 2, 2, 1]
    region_modality.standard_deviations = [20.0, 7.0, 3.0, 1.5]
    
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
    
    # Load the GT poses (initial pose + poses for each image)
    body2world_poses_gt = np.empty((config.nb_images+1, 4, 4), dtype=np.float32)
    
    if config.dataset_name == "rbot":
        # Load the initial body2world GT pose
        poses_file = Path(config.dataset_dir) / "poses_first.txt"
        
        with open(poses_file, "r") as f:
            f.readline()  # Skip the first line
            lines = f.readlines()
        
        for i in range(body2world_poses_gt.shape[0]):
            
            # Parse the pose components
            pose_components = list(map(float, lines[i].strip("\n").split("\t")))
            # Create the homogenous transformation matrix as a list of lists
            body2world_poses_gt[i] = [
                pose_components[:3] + [pose_components[9]*1e-3],
                pose_components[3:6] + [pose_components[10]*1e-3],
                pose_components[6:9] + [pose_components[11]*1e-3],
                [0, 0, 0, 1],
            ]
    
    # Create the initial body2world pose object
    body2world_pose = pym3t.Transform3fA(
        np.array(body2world_poses_gt[0], dtype=np.float32)
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
        model=config.model,
        scene=config.scene,
        nb_images=config.nb_images,
        execute_detection_and_start_tracking=True,
        display_images=config.display_images,
        stop_at_each_image=True,
        body2world_poses_gt=body2world_poses_gt,
        body=body,
        reset_pose=config.reset_pose,
        log_dir=Path(config.log_dir),
    ):
        return -1
    
    return 0


if __name__ == "__main__":
    # Configuration file path
    config_path = Path("config/track_example_rbot.yaml")

    # Read config file
    config = omegaconf.OmegaConf.load(config_path)
    
    # Create the log directory
    log_dir =\
        Path(config.log_dir) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Update the log directory in the configuration
    config.log_dir = str(log_dir)
    
    # Run tracking example
    track(config)
