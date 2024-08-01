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
        ValueError: If the modality is not recognized.
    
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

    # Load camera intrinsics depending on the dataset
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
        image_name_pre = ""
        n_leading_zeros = 4
        load_directory = Path(config.dataset_dir) / config.scene / config.model
        
        # Load camera intrinsics
        camera_intrinsics_file = Path(config.dataset_dir) / config.scene / "K.txt"
        with open(camera_intrinsics_file, "r") as f:
            line = f.readline()
        # Parse intrinsics
        K_flat = list(map(float, line[line.find("(")+1:line.find(")")].split(", ")))
        # Create intrinsics dictionary
        intrinsics = {
            "fu": float(K_flat[0]),
            "fv": float(K_flat[4]),
            "ppu": float(K_flat[2]),
            "ppv": float(K_flat[5]),
            "width": config.image_size.width,
            "height": config.image_size.height,
        }
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
    
    # Get the path to the 3D model depending on the dataset
    if config.dataset_name == "rbot":
        geometry_path =\
            Path(config.dataset_dir) / config.model / f"{config.model}.obj"
    elif config.dataset_name == "bcot":
        geometry_path =\
            Path(config.dataset_dir) / f"models/{config.model}.obj"
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

    # Create a directory to save the region model if it does not exist
    region_model_dir = Path("tmp") / config.dataset_name
    region_model_dir.mkdir(parents=True, exist_ok=True)
    # Set up region model
    region_model = pym3t.RegionModel(
        name=f"{config.model}_region_model",
        body=body,
        model_path=f"{region_model_dir}/{config.model}_region_model.bin",
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
    # Region checking consists in discarding lines crossing more than 1 time
    # the frontier between the two regions
    if use_region_checking:
        silhouette_renderer = pym3t.FocusedSilhouetteRenderer(
            name=f"{config.model}_silhouette_renderer",
            renderer_geometry=renderer_geometry,
            world2camera_pose=pym3t.Transform3fA(np.eye(4, dtype=np.float32)),
            intrinsics=intrinsics,
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
    
    if config.nb_images is None:
        if config.dataset_name == "rbot":
            # Constant number of images for the RBOT dataset
            # (do not take the first image into account)
            config.nb_images = 1001 - 1
        elif config.dataset_name == "bcot":
            # Get the number of images for the current scene/model
            # (do not take the first image into account)
            config.nb_images = len(list(Path(load_directory).glob("*.png"))) - 1
        else:
            raise ValueError(f"Unknown dataset name: {config.dataset_name}")
    
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
    elif config.dataset_name == "bcot":
        # Load the initial body2world GT pose
        poses_file = Path(config.dataset_dir) / config.scene / config.model / "pose.txt"
        
        with open(poses_file, "r") as f:
            lines = f.readlines()
        
        for i in range(body2world_poses_gt.shape[0]):
            
            # Parse the pose components
            pose_components = list(map(float, lines[i].strip("\t\n").split("\t")))
            # Create the homogenous transformation matrix as a list of lists
            body2world_poses_gt[i] = [
                pose_components[:3] + [pose_components[9]*1e-3],
                pose_components[3:6] + [pose_components[10]*1e-3],
                pose_components[6:9] + [pose_components[11]*1e-3],
                [0, 0, 0, 1],
            ]
    else:
        raise ValueError(f"Unknown dataset name: {config.dataset_name}")
    
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
    config_path = Path("config/track_example_bcot.yaml")

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
