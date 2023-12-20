from pathlib import Path
import rich
from rich import print
import os
import sys
import torch


from nerfstudio.criticalpixel.data.dataparser.colmap_parser import ColmapDataparserConfig

config = ColmapDataparserConfig(data_dir="/data/lidong/data/nerfuser/A")
from rich.traceback import install

install(show_locals=False)

parser = config.setup()
parser.parse_data()
