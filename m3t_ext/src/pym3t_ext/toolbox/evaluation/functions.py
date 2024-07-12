# Standard libraries
from pathlib import Path


def load_all_scenes_models(
    dataset_name: str,
    dataset_dir: Path,
) -> dict:
    
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
        raise NotImplementedError("Evaluation for BCOT dataset is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
