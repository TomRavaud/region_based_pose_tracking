# Standard libraries
from pathlib import Path
import datetime

# Third-party libraries
import omegaconf

# Custom modules and libraries
import pym3t_ext.toolbox.utils.scenes_models as scenes_models
from pym3t_ext.scripts.track import track


def evaluate(config: omegaconf.DictConfig) -> None:
    """Entry point for the evaluation script.

    Args:
        config (omegaconf.DictConfig): Configuration parameters.
    """
    if "scenes_models_dict" in config:
        if config.scenes_models_dict == "all":
            scenes_models_dict = scenes_models.load_all_scenes_models(
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
    
    for scene, models in scenes_models_dict.items():
        for model in models:
            print(f"Scene: {scene}, Model: {model}")
            
            # Set the configuration parameters
            track_config = omegaconf.OmegaConf.create({
                "dataset_name": config.dataset_name,
                "dataset_dir": config.dataset_dir,
                "model": model,
                "scene": scene,
                "image_size": config.image_size,
                "geometry_unit_in_meter": config.geometry_unit_in_meter,
                "modality": config.modality,
                "reset_pose": config.reset_pose,
                "nb_images": config.nb_images_per_sequence,
                "display_images": False,
                "log_dir": config.log_dir,
            })
            
            # Run tracking
            track(track_config)


if __name__ == "__main__":
    # Configuration file path
    config_path = Path("config/evaluate_bcot.yaml")

    # Read config file
    config = omegaconf.OmegaConf.load(config_path)
    
    # Create the log directory
    log_dir =\
        Path(config.log_dir) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Update the log directory in the configuration
    config.log_dir = str(log_dir)
 
    # Run evaluation
    evaluate(config)
