# Standard libraries
from pathlib import Path


def load_all_scenes_models(
    dataset_name: str,
    dataset_dir: Path,
) -> dict:
    """Create a dictionary containing all scenes and models for a given dataset.

    Args:
        dataset_name (str): Name of the dataset.
        dataset_dir (Path): Path to the dataset directory.

    Raises:
        ValueError: If the dataset name is unknown.

    Returns:
        dict: Dictionary containing all scenes and models for the dataset.
    """
    if dataset_name == "rbot":
        models = [f.name for f in dataset_dir.iterdir() if f.is_dir()]
        scenes = [
            "a_regular",
            "b_dynamiclight",
            "c_noisy",
            "d_occlusion",
        ]
        return {scene: [model for model in models] for scene in scenes}
    
    elif dataset_name == "bcot":
        scenes = [scene for scene in dataset_dir.iterdir() if scene.is_dir()
                  and scene.name != "models"]
        return {
            scene.name: [model.name for model in scene.iterdir() if model.is_dir()]
            for scene in scenes
        }
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
