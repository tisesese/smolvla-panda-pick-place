
"""Repository-relative defaults; external data and checkpoints are opt-in."""
import os
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
def location(name,default):return Path(os.environ.get(name,str(default))).expanduser().resolve()
OLD=location('TASK4_ASSETS',PROJECT_ROOT/'simulation')
OUTPUT=location('TASK4_OUTPUT',PROJECT_ROOT/'runs')
DATA=location('TASK4_DATA',PROJECT_ROOT/'data/lerobot_panda_pick_place_100_lowdisk')
EVIDENCE=location('TASK4_EVIDENCE',PROJECT_ROOT/'data_metadata/panda_100_lowdisk')
BASE=location('TASK4_BASE',PROJECT_ROOT/'model_bundle/base/pretrained_model')
SELECTED=location('TASK4_MODEL',PROJECT_ROOT/'model_bundle/033000/pretrained_model')
VLM_CONFIG=location('TASK4_VLM_CONFIG',PROJECT_ROOT/'assets/vlm')
SCENE=OLD/'experiments/panda_pick_place/scene/panda_pick_place.xml'
TASK='Pick up the red cube and place it in the tray.'
IMAGE='observation.images.fixed_rgb'
SEED=20261023
FPS=50
CHUNK=50
