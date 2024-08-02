# Standard libraries
from abc import ABC, abstractmethod
from pathlib import Path

# Third-party libraries
import numpy as np


class Metric(ABC):
    """
    Abstract base class for metrics.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: Name of the metric.
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict:
        """Get the parameters of the metric.

        Returns:
            dict: Parameters of the metric.
        """
        pass
    
    @abstractmethod
    def __call__(self, T_gt: np.ndarray, T_est: np.ndarray) -> float:
        """Compute the metric between two poses.

        Args:
            T_gt (np.ndarray): Ground truth pose.
            T_est (np.ndarray): Estimated pose.

        Returns:
            float: Metric value.
        """
        pass


class CmDegreeScore(Metric):
    """
    Check if a translation and rotation errors are jointly below given thresholds.
    """
    def __init__(self, threshold_trans: float = 5., threshold_rot: float = 5.) -> None:
        """Initialize the parameters of the metric.

        Args:
            threshold_trans (float, optional): Threshold on translation error in cm.
                Defaults to 5.
            threshold_rot (float, optional): Threshold on rotation error in degrees.
                Defaults to 5.
        """
        self._threshold_trans = threshold_trans
        self._threshold_rot = threshold_rot
    
    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: Name of the metric.
        """
        threshold_trans = self._threshold_trans
        threshold_rot = self._threshold_rot
        
        # If the threshold is an integer, remove the decimal part
        if self._threshold_trans == int(self._threshold_trans):
            threshold_trans = int(self._threshold_trans)
        if self._threshold_rot == int(self._threshold_rot):
            threshold_rot = int(self._threshold_rot)
        
        return f"{threshold_trans}cm_{threshold_rot}deg"
        
    @property
    def parameters(self) -> dict:
        """Get the parameters of the metric.

        Returns:
            dict: Parameters of the metric.
        """
        return {
            "threshold_trans": self._threshold_trans,
            "threshold_rot": self._threshold_rot,
        }
    
    def __call__(self, T_gt: np.ndarray, T_est: np.ndarray) -> float:
        """Compute the translation and rotation errors between two poses and check
        if they are below a given threshold.

        Args:
            T_gt (np.ndarray): Ground truth pose.
            T_est (np.ndarray): Estimated pose.

        Returns:
            float: Success.
        """
        # Get rotation matrices and translation vectors
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]
        R_est = T_est[:3, :3]
        t_est = T_est[:3, 3]

        # Translation error
        error_trans = np.linalg.norm(t_gt - t_est)

        # Rotation error
        arg = (np.trace(np.dot(R_est.T, R_gt)) - 1) / 2
        # Check if the argument is out of bounds (due to numerical errors)
        if arg > 1:
            error_rot = 0
        elif arg < -1:
            error_rot = np.pi
        # Compute the error
        else:
            error_rot = np.arccos((np.trace(np.dot(R_est.T, R_gt)) - 1) / 2)

        # Convert the translation threshold to meters
        threshold_trans = self._threshold_trans * 1e-2

        # Convert the rotation threshold to radians
        threshold_rot = self._threshold_rot * np.pi / 180

        # Check if the error is below the threshold
        success = error_trans < threshold_trans and error_rot < threshold_rot

        return float(success)


class VerticesMetric(Metric, ABC):
    """
    Abstract base class for metrics that use the vertices of the model.
    """
    def __init__(self, model_file: Path) -> None:
        """Initialize the metric with the model file.

        Args:
            model_file (Path): Path to the model file (e.g. .obj).
        
        Raises:
            ValueError: If the model file format is unknown.
        """
        self._model_file = model_file
        self._load_vertices()
        self._compute_model_diameter()
        
    # @property
    # def model_file(self) -> Path:
    #     """Get the path to the model file.

    #     Returns:
    #         Path: Path to the model file.
    #     """
    #     return self._model_file
    
    # @property
    # def model_name(self) -> str:
    #     """Get the name of the model.

    #     Returns:
    #         str: Name of the model.
    #     """
    #     return self._model_file.stem
    
    
    def _load_vertices(self) -> None:
        """Load the vertices of the model."""
        # Extract the vertices from the model
        if self._model_file.suffix == ".obj":
            self._vertices = self._vertices_from_obj(self._model_file)
        else:
            raise ValueError(f"Unknown model file format: {self._model_file.suffix}")
    
    @staticmethod
    def _vertices_from_obj(file_path: Path) -> np.ndarray:
        """Read an .obj file and extract the vertices.

        Args:
            file_path (Path): Path to the model file.

        Returns:
            np.ndarray: Vertices.
        """
        vertices = []

        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.strip().split(' ')[1:]])

        return np.array(vertices)

    @property
    def vertices(self) -> np.ndarray:
        """Get the vertices of the model.

        Returns:
            np.ndarray: Vertices of the model.
        """
        if self._vertices is None:
            print("Vertices not loaded yet.")
        return self._vertices
    
    @staticmethod
    def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Transform the points using the given pose.

        Args:
            T (np.ndarray): Homogeneous transformation matrix.
            points (np.ndarray): Points to transform. Shape (N, 3).

        Returns:
            np.ndarray: Transformed points.
        """
        return np.dot(T[:3, :3], points.T).T + T[:3, 3]
    
    # TODO: check what diameter is used in DeepAC
    def _compute_model_diameter(self) -> float:
        """Compute the diameter of the model.

        Returns:
            float: Diameter of the model.
        """
        # Distance between all pairs of vertices
        distances = np.linalg.norm(self._vertices[:, None] - self._vertices, axis=2)
        self._diameter = np.max(distances)


class ADD(VerticesMetric):
    """
    Compute the average distance between transformed vertices.
    """
    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: Name of the metric.
        """
        return "ADD"
    
    @property
    def parameters(self) -> dict:
        """Get the parameters of the metric.

        Returns:
            dict: Parameters of the metric.
        """
        return {
            "model_file": self._model_file,
        }
    
    def __call__(self, T_gt: np.ndarray, T_est: np.ndarray) -> float:
        """Compute the metric between two poses.

        Args:
            T_gt (np.ndarray): Ground truth pose.
            T_est (np.ndarray): Estimated pose.

        Returns:
            float: Metric value.
        """
        # Transform the vertices of the model
        vertices_gt = self._transform_points(T_gt, self._vertices)
        vertices_est = self._transform_points(T_est, self._vertices)
        
        # Compute the distances between the transformed vertices
        distances = np.linalg.norm(vertices_gt - vertices_est, axis=1)
        
        return np.mean(distances)


class ADDs(VerticesMetric):
    """
    Variant of the ADD metric suited for symmetric (s) objects. Compute the average
    distance to the closest model point.
    """
    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: Name of the metric.
        """
        return "ADDs"
    
    @property
    def parameters(self) -> dict:
        """Get the parameters of the metric.

        Returns:
            dict: Parameters of the metric.
        """
        return {
            "model_file": self._model_file,
        }
    
    def __call__(self, T_gt: np.ndarray, T_est: np.ndarray) -> float:
        """Compute the metric between two poses.

        Args:
            T_gt (np.ndarray): Ground truth pose.
            T_est (np.ndarray): Estimated pose.

        Returns:
            float: Metric value.
        """
        # Transform the vertices of the model
        vertices_gt = self._transform_points(T_gt, self._vertices)
        vertices_est = self._transform_points(T_est, self._vertices)
        
        # Compute the distances between all pairs of vertices
        distances = np.linalg.norm(
            vertices_gt[:, None] - vertices_est,
            axis=2,
        )
        # For each ground truth vertex, get the distance to the closest estimated vertex
        # and compute the average
        return np.mean(np.min(distances, axis=1))


class ADDkd(ADD):
    
    def __init__(self, model_file: Path, k: float) -> None:
        """Initialize the parameters of the metric.

        Args:
            model_file (Path): Path to the model file.
            k (float): Scale factor.
        """
        super().__init__(model_file)
        self._k = k
    
    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: Name of the metric.
        """
        k = self._k
        if self._k == int(self._k):
            k = int(self._k)
        return f"ADD_{k}d"
    
    @property
    def parameters(self) -> dict:
        """Get the parameters of the metric.

        Returns:
            dict: Parameters of the metric.
        """
        return {**super().parameters, "k": self._k}
    
    def __call__(self, T_gt: np.ndarray, T_est: np.ndarray) -> float:
        """Compute the metric between two poses.

        Args:
            T_gt (np.ndarray): Ground truth pose.
            T_est (np.ndarray): Estimated pose.

        Returns:
            float: Metric value.
        """
        # Compute the ADD metric
        add_score = super().__call__(T_gt, T_est)
        
        return float(add_score < self._k * self._diameter)

class ADDskd(ADDs):
    
    def __init__(self, model_file: Path, k: float) -> None:
        """Initialize the parameters of the metric.

        Args:
            model_file (Path): Path to the model file.
            k (float): Scale factor.
        """
        super().__init__(model_file)
        self._k = k
    
    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: Name of the metric.
        """
        k = self._k
        if self._k == int(self._k):
            k = int(self._k)
        return f"ADDs_{k}d"
    
    @property
    def parameters(self) -> dict:
        """Get the parameters of the metric.

        Returns:
            dict: Parameters of the metric.
        """
        return {**super().parameters, "k": self._k}
    
    def __call__(self, T_gt: np.ndarray, T_est: np.ndarray) -> float:
        """Compute the metric between two poses.

        Args:
            T_gt (np.ndarray): Ground truth pose.
            T_est (np.ndarray): Estimated pose.

        Returns:
            float: Metric value.
        """
        # Compute the ADDs metric
        adds_score = super().__call__(T_gt, T_est)
        
        return float(adds_score < self._k * self._diameter)


if __name__ == "__main__":
    pass
