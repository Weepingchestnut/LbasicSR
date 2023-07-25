import os
from typing import Dict, List

import yaml

from lbasicsr.utils.options import yaml_load


def change_yaml(yaml_set: Dict, scale) -> None:
    yaml_dict = yaml_load(yaml_set["path"])
    
    # change need setting -------------------------------------
    # name
    yaml_dict["name"] = yaml_set["name"] + f"_x{str(scale)}"
    
    # scale
    if isinstance(scale, tuple):
        scale = scale
    else:
        scale = (scale, scale)

    yaml_dict["scale"] = scale
    # yaml_dict["datasets"]["test_1"]["downsampling_scale"] = scale

    if "Bicubic" in yaml_dict["name"]:
        yaml_dict["network_g"]["scale"] = scale
    
    with open(yaml_set["path"], "w") as f:
        yaml.dump(yaml_dict, stream=f)
    
    # print(yaml_dict)


def run_python(yaml_set: Dict, scales: List, cuda_id: int=0) -> None:
    cuda_visible_devices = "CUDA_VISIBLE_DEVICES={}".format(cuda_id)
    
    for scale in scales:
        change_yaml(yaml_set, scale)

        cmd = f"{cuda_visible_devices} " + \
            "python -u lbasicsr/test.py -opt " + yaml_set["path"]
        
        # print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    yaml_set = {
        "path": "options/test/DUF/bash_run/test_DUF_train_x4_Vid4_asBI.yml",
        "name": "EDVR_L_x2_transfer_UDM10_SR",
    }
    
    # scales = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
    #           2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
    #           3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4,
    #           (1.5, 4), (2, 4), (1.5, 3.5), (1.6, 3.05), (3.5, 2), (3.5, 1.75), (4, 1.4)]
    # scales = [(1.5, 4), (2, 4), (1.5, 3.5), (1.6, 3.05), (3.5, 2), (3.5, 1.75), (4, 1.4)]
    
    # scales = [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
    # scales = [1.1]
    scales = [(2, 3.75), (1.7, 3.75), (2.95, 3.75), (3.9, 2), (3.5, 1.5)]

    run_python(yaml_set, scales, cuda_id=0)

