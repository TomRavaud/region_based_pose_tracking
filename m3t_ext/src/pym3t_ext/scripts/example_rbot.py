# Third-party libraries
import numpy as np
import pym3t
from pym3t_ext.components.pytracker import PyTracker
from pym3t_ext.components.deep_region_modality import DeepRegionModality


DATASET_PATH = "data/RBOT/"
OBJECT_NAME = "ape"
GEOMETRY_UNIT_IN_METER = 0.001
INTRINSICS = pym3t.Intrinsics(  # GT intrinsics (cf dataset)
    fu=650.048,
    fv=647.183,
    ppu=324.328,
    ppv=257.323,
    width=640,
    height=512,
)
BODY2WORLD_POSE = pym3t.Transform3fA(  # GT pose (cf dataset)
    np.array([
        [0.997056, -0.04307, 0.0634383, 0.019051],
        [0.043157, 0.999068, 0, 0.00408901],
        [-0.0633792, 0.0027378, 0.997986, 0.549],
        [0, 0, 0, 1]
    ], dtype=np.float32)
)


def main():
    
    # Set up tracker
    tracker = PyTracker(
        name="tracker",
        synchronize_cameras=False,
    )

    # Set up renderer geometry
    renderer_geometry = pym3t.RendererGeometry(
        name="renderer_geometry",
    )

    # Set up camera
    color_camera = pym3t.LoaderColorCamera(
        name="color_camera",
        load_directory=f"{DATASET_PATH}{OBJECT_NAME}/frames/",
        intrinsics=INTRINSICS,
        image_name_pre="a_regular",
        load_index=0,
        n_leading_zeros=4,
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
        name=OBJECT_NAME,
        geometry_path=f"{DATASET_PATH}{OBJECT_NAME}/{OBJECT_NAME}.obj",
        geometry_unit_in_meter=GEOMETRY_UNIT_IN_METER,
        geometry_counterclockwise=True,
        geometry_enable_culling=True,
        geometry2body_pose=pym3t.Transform3fA(np.eye(4, dtype=np.float32)),
    )
    renderer_geometry.AddBody(body)

    # Set up region model
    region_model = pym3t.RegionModel(
        name=f"{OBJECT_NAME}_region_model",
        body=body,
        model_path=f"tmp/{OBJECT_NAME}_region_model.bin",
    )
    
    # Set up region modality
    # region_modality = pym3t.RegionModality(
    #     name=f"{OBJECT_NAME}_region_modality",
    #     body=body,
    #     color_camera=color_camera,
    #     region_model=region_model,
    # )
    # # Set visualization options
    # region_modality.visualize_lines_correspondence = False
    # region_modality.visualize_points_optimization = False

    # Set up deep region modality
    region_modality = DeepRegionModality(
        name=f"{OBJECT_NAME}_deep_region_modality",
        body=body,
        color_camera=color_camera,
        region_model=region_model,
    )
    
    # Set up link
    link = pym3t.Link(
        name=f"{OBJECT_NAME}_link",
        body=body,
    )
    link.AddModality(region_modality)
    
    # Set up optimizer
    optimizer = pym3t.Optimizer(
        name=f"{OBJECT_NAME}_optimizer",
        root_link=link,
    )
    tracker.AddOptimizer(optimizer) 
    
    # Set up detector
    detector = pym3t.StaticDetector(
        name="detector",
        optimizer=optimizer,
        link2world_pose=BODY2WORLD_POSE,
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
    
    main()
