# Standard libraries
from pathlib import Path
from functools import partial

# Third-party libraries
import omegaconf
import numpy as np
from tabulate import tabulate

# Custom modules
from pym3t_ext.toolbox.geometry.metrics import (
    cm_degree_score,
)


def poses_txt_to_np(file_path: Path, scale_trans: float = 1.0) -> np.ndarray:
    """Read the poses from a text file and return them as a numpy array.

    Args:
        file_path (Path): Path to the text file containing the poses.
        scale_trans (float, optional): Scale factor for the translation.
            Defaults to 1.0.

    Returns:
        np.ndarray: Poses as a numpy array.
    """
    with open(file_path, "r") as file:
        # Discard the first line (header)
        file.readline()
        lines = file.readlines()
    
    # Parse the lines
    lines_np = np.array([list(map(float, line.strip().split())) for line in lines])
    poses = np.array([np.array([
        [lines_np[i, 0], lines_np[i, 1], lines_np[i, 2], lines_np[i, 9]*scale_trans],
        [lines_np[i, 3], lines_np[i, 4], lines_np[i, 5], lines_np[i, 10]*scale_trans],
        [lines_np[i, 6], lines_np[i, 7], lines_np[i, 8], lines_np[i, 11]*scale_trans],
        [0, 0, 0, 1],
    ]) for i in range(len(lines_np))])
    
    return poses


def scores_to_table(scores_dict: dict) -> str:
    """Create a table from a scores dictionary.

    Args:
        scores_dict (dict): Dictionary containing the scores.
            The dictionary should have the following structure:
            {
                scene1: {
                    object1: score1,
                    object2: score2,
                    ...
                },
                scene2: {
                    object1: score1,
                    object2: score2,
                    ...
                },
                ...
            }

    Returns:
        str: Table as a string.
    """
    # Extract object names from the data
    objects = list(next(iter(scores_dict.values())).keys())

    # Header of the table
    header = ["Scene \ Object"] + objects + ["Average"]

    # Create the rows for each scene and add an average column at the end
    rows = []
    for scene, scores in scores_dict.items():
        row = [scene] + [scores[obj] for obj in objects] +\
            [np.mean(list(scores.values()))]
        rows.append(row)

    # Add a total row
    rows.append(["Total"] + [""]*(len(header) - 2) +\
        [np.mean([row[-1] for row in rows])])
    
    # Create the table
    table = tabulate(rows, headers=header, tablefmt="fancy_grid")
    
    return table


def compute_metrics(config: omegaconf.DictConfig) -> None:
    """Compute the metrics for the given configuration.

    Args:
        config (omegaconf.DictConfig): Configuration parameters.

    Raises:
        ValueError: If the metric is unknown.
        ValueError: _description_
    """
    if config.dataset_name == "rbot":
        # GT poses are stored in one file, read it
        poses_gt = poses_txt_to_np(
            Path(config.dataset_dir) / "poses_first.txt",
            scale_trans=1e-3,
        )
        if config.skip_first_frame:
            poses_gt = poses_gt[1:]
    
    # Get the metrics to compute and set their parameters
    metrics = {}
    for metric in config.metrics:
        metric_type = list(metric.keys())[0]
        metric_name = metric[metric_type]["name"] if "name" in metric[metric_type]\
            else metric_type
        
        if metric_type == "cm_degree_score":
            metrics[metric_name] = partial(
                cm_degree_score,
                threshold_trans=metric[metric_type]["threshold_trans"],
                threshold_rot=metric[metric_type]["threshold_rot"],
            )
        else:
            raise ValueError(f"Unknown metric: {metric_type}")
    
    # Create dictionaries to store the results per scene
    scores_dicts = [{} for _ in range(len(metrics))]
    
    # Get the log files paths
    experiment_dir = Path(config.log_dir) / config.experiment_name
    log_files = sorted(experiment_dir.glob("*.txt"))
    
    for log_file in log_files:
        
        # Lists to store the results for each frame before averaging
        scores_lists = [[] for _ in range(len(metrics))]
        
        # Get the model and scene names
        if config.dataset_name in ["rbot", "bcot"]:
            model = log_file.stem.split("_")[1]
            scene = "_".join(log_file.stem.split("_")[2:])
        
        # Read the poses
        poses_est = poses_txt_to_np(log_file)
        if config.skip_first_frame:
            poses_est = poses_est[1:]
        
        if poses_gt.shape != poses_est.shape:
            raise ValueError("GT and estimated poses have different shapes.")
        
        # Compute the metrics
        for T_gt, T_est in zip(poses_gt, poses_est):
            for i, metric in enumerate(metrics.values()):
                scores_lists[i].append(metric(T_gt, T_est))
        
        # Add the score associated to scene s and object o to the data dictionary
        if scene not in scores_dicts[0]:
            # Create an empty dictionary for the scene s for each metric
            for scores_dict in scores_dicts:
                scores_dict[scene] = {}

        for scores_dict, scores_list in zip(scores_dicts, scores_lists):
            scores_dict[scene][model] = np.mean(scores_list)
            
    tables = [scores_to_table(scores_dict) for scores_dict in scores_dicts]
    
    # Save the tables to a file
    for metric_name, table in zip(metrics.keys(), tables):
        with open(result_dir / f"{config.dataset_name}_{metric_name}.txt", "w") as f:
            f.write(table)


if __name__ == "__main__":
    
    # Configuration file path
    config_path = Path("config/compute_metrics.yaml")

    # Read config file
    config = omegaconf.OmegaConf.load(config_path)
    
    # Create the results directory
    result_dir =\
        Path(config.result_dir) / config.experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Update the results directory in the configuration
    config.result_dir = str(result_dir)
    
    # Run tracking example
    compute_metrics(config)
