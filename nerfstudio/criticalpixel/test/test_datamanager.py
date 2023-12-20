from pathlib import Path
import rich
from rich import print
import os
import sys


from nerfstudio.criticalpixel.data.dataparser.colmap_parser import ColmapDataparserConfig
from nerfstudio.criticalpixel.data.datamanager.nerf_data_manager import NeRFDataManagerConfig
from nerfstudio.criticalpixel.data.datamanager.data_partation import TrainFullDataPartationConfig
from rich.progress import track

data_manager_config = NeRFDataManagerConfig(
    dataparser=ColmapDataparserConfig(data_dir="/data/lidong/data/nerfuser/A"),
    spliter=TrainFullDataPartationConfig(),
    num_image_to_load=200,
    num_iter_to_resample=1000,
)

from rich.traceback import install

install(show_locals=False)

parser = data_manager_config.setup(device="cuda")


for i in track(range(1000), total=1000, console=rich.console.Console()):
    data = parser.next_train(i)

for i in track(range(1000), total=1000):
    data = parser.next_eval(i)

for i in track(range(1000), total=1000):
    data = parser.next_eval_image(i)
