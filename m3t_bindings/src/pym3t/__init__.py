# from ._pym3t_mod import Tracker
# from ._pym3t_mod import RendererGeometry
# from ._pym3t_mod import LoaderColorCamera
# from ._pym3t_mod import Intrinsics, IDType
# from ._pym3t_mod import DummyColorCamera, DummyDepthCamera
# from ._pym3t_mod import NormalColorViewer
# from ._pym3t_mod import FocusedBasicDepthRenderer, FocusedSilhouetteRenderer
# from ._pym3t_mod import Body
# from ._pym3t_mod import Link
# from ._pym3t_mod import StaticDetector
# from ._pym3t_mod import RegionModel, DepthModel
# from ._pym3t_mod import RegionModality, DepthModality, TextureModality
# from ._pym3t_mod import Optimizer
# from ._pym3t_mod import WITH_REALSENSE

from ._pym3t_mod import *

# if WITH_REALSENSE:
#     from ._pym3t_mod import RealSenseColorCamera, RealSenseDepthCamera

# Selectively import the classes to be exposed to the user
# __all__ = [
#     'Tracker', 
#     'RendererGeometry',
#     'Intrinsics', 'IDType',
#     'DummyColorCamera', 'DummyDepthCamera', 
#     'NormalColorViewer', 'NormalDepthViewer', 
#     'FocusedBasicDepthRenderer', 'FocusedSilhouetteRenderer'
#     'Body', 'Link', 
#     'StaticDetector', 
#     'RegionModel', 'DepthModel', 
#     'RegionModality', 'DepthModality', 'TextureModality' 
#     'Optimizer',
# ] 

# if WITH_REALSENSE:
#     __all__.append(['RealSenseColorCamera', 'RealSenseDepthCamera'])