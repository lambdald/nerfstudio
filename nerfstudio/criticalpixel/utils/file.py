"""
Description: 文件相关
"""
from typing import List, Union
from pathlib import Path
import shutil
import zipfile
from zipfile import ZipFile
import pickle
import io
import os

import numpy as np
import torch


def get_all_files(root: Union[str, Path], pattern: str) -> List[Path]:
    """get files in root dir, including files in subdirs.

    Args:
        root (str): _description_
        pattern (str): _description_

    Returns:
        List[Path]: _description_
    """
    root = Path(root)
    files = []
    for img_path in root.glob(pattern):
        files.append(img_path)

    for subdir in root.iterdir():
        if subdir.is_dir():
            files += get_all_files(subdir, pattern)
    return sorted(files)


def glob_imgs(path) -> List[Path]:
    imgs = []
    for ext in ["*.png", "*.PNG", "*.jpeg", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(get_all_files(path, ext))
    return sorted(imgs)


def create_dir_if_not_exists(dir_path):
    lpath = Path(dir_path)
    if not lpath.exists():
        print("create dir:", dir_path)
        lpath.mkdir(parents=True)


def delete_if_exists(inpath):
    lpath = Path(inpath)
    if not lpath.exists():
        return

    if lpath.is_dir():
        shutil.rmtree(lpath)
    else:
        lpath.unlink()


def save_data_by_zip(file_path, data, data_type):
    file_path = Path(file_path)
    with ZipFile(str(file_path), compression=zipfile.ZIP_DEFLATED, mode="w") as zf:
        with zf.open(file_path.name, "w") as f:
            if data_type == "np":
                np.save(f, data)
            elif data_type == "pickle":
                pickle.dump(data, f)
            elif data_type == "torch":
                torch.save(data, f)
            else:
                raise NotImplementedError


def load_data_by_zip(file_path, data_type):
    file_path = Path(file_path)

    with ZipFile(str(file_path)) as zf:
        with zf.open(file_path.name) as f:
            buffer = io.BytesIO(f.read())
            if data_type == "np":
                data = np.load(buffer)
            elif data_type == "pickle":
                data = pickle.load(buffer)
            elif data_type == "torch":
                data = torch.load(f)
            else:
                raise NotImplementedError
    return data


def is_empty_dir(path):
    return len(os.listdir(path)) == 0
