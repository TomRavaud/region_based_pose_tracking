# Standard libraries
from pathlib import Path

# Third-party libraries
import omegaconf

# Custom modules and libraries
import pym3t_ext.toolbox.evaluation.functions as eval
from pym3t_ext.scripts.track import track


def evaluate(config: omegaconf.DictConfig) -> None:
    """Entry point for the evaluation script.

    Args:
        config (omegaconf.DictConfig): Configuration parameters.
    """
    if "scenes_models_dict" in config:
        if config.scenes_models_dict == "all":
            scenes_models_dict = eval.load_all_scenes_models(
                dataset_name=config.dataset_name,
                dataset_dir=Path(config.dataset_dir),
            )
        elif isinstance(config.scenes_models_dict, omegaconf.DictConfig):
            scenes_models_dict = dict(config.scenes_models_dict)
        else:
            raise ValueError(
                "The 'scenes_models_dict' key must be either 'all' or a dictionary."
            )
    else:
        raise ValueError(
            "The 'scenes_models_dict' key is missing in the configuration file."
        )
    
    if config.dataset_name == "rbot":
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
        
        # Load the initial body2world GT pose
        poses_file = Path(config.dataset_dir) / "poses_first.txt"
        with open(poses_file, "r") as f:
            f.readline()  # Skip the first line
            line = f.readline()
        # Parse the pose components
        pose_components = list(map(float, line.strip("\n").split("\t")))
        # Create the homogenous transformation matrix as a list of lists
        body2world_pose = [
            pose_components[:3] + [pose_components[9]*1e-3],
            pose_components[3:6] + [pose_components[10]*1e-3],
            pose_components[6:9] + [pose_components[11]*1e-3],
            [0, 0, 0, 1],
        ]
 
    for scene, models in scenes_models_dict.items():
        for model in models:
            print(f"Scene: {scene}, Model: {model}")
            
            # Set the configuration parameters
            track_config = omegaconf.OmegaConf.create({
                "dataset_name": config.dataset_name,
                "dataset_dir": config.dataset_dir,
                "model": model,
                "scene": scene,
                "geometry_unit_in_meter": config.geometry_unit_in_meter,
                "intrinsics": intrinsics,
                "body2world_pose": body2world_pose,
            })
            
            # Run tracking
            track(track_config)



if __name__ == "__main__":
    # Configuration file path
    config_path = Path("config/evaluate_rbot.yaml")

    # Read config file
    config = omegaconf.OmegaConf.load(config_path)
 
    # Run evaluation
    evaluate(config)
