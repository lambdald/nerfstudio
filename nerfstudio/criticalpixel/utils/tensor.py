import torch
from typing import Union, Optional, List, Dict


def move_tensor_to_device(
    stuff: Union[Dict, torch.Tensor], device: Union[torch.device, str] = "cpu", exclude: Optional[List[str]] = None
) -> Dict:
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = move_tensor_to_device(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    return stuff


def collate_list_of_dict(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(data) == 0:
        return {}
    collated_data = {}
    for key in data[0].keys():
        collated_data[key] = [d[key] for d in data]
        if isinstance(collated_data[key][0], torch.Tensor):
            collated_data[key] = torch.cat(collated_data[key], dim=0)
    return collated_data
