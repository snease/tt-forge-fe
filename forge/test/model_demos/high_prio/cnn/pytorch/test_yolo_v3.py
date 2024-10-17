# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys

from PIL import Image

import torch
import forge

# https://github.com/holli/yolov3_pytorch
sys.path = list(set(sys.path + ["third_party/confidential_customer_models/model_2/pytorch/"]))

from yolo_v3.holli_src import utils
from yolo_v3.holli_src.yolo_layer import *
from yolo_v3.holli_src.yolov3_tiny import *
from yolo_v3.holli_src.yolov3 import *


def generate_model_yolotinyV3_imgcls_holli_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object

    model = Yolov3Tiny(num_classes=80, use_wrong_previous_anchors=True)
    model.load_state_dict(
        torch.load("third_party/confidential_customer_models/model_2/pytorch/yolo_v3/weights/yolov3_tiny_coco_01.h5")
    )
    model.eval()

    sz = 512
    imgfile = "third_party/confidential_customer_models/model_2/pytorch/yolo_v3/person.jpg"
    img_org = Image.open(imgfile).convert("RGB")
    img_resized = img_org.resize((sz, sz))
    img_tensor = utils.image2torch(img_resized)

    return model, [img_tensor], {}


def test_yolov3_tiny_holli_pytorch(test_device):
    model, inputs, _ = generate_model_yolotinyV3_imgcls_holli_pytorch(
        test_device,
        None,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]])


def generate_model_yoloV3_imgcls_holli_pytorch(test_device, variant):
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    model = Yolov3(num_classes=80)
    model.load_state_dict(
        torch.load(
            "third_party/confidential_customer_models/model_2/pytorch/yolo_v3/weights/yolov3_coco_01.h5",
            map_location=torch.device("cpu"),
        )
    )
    model.eval()

    sz = 512
    imgfile = "third_party/confidential_customer_models/model_2/pytorch/yolo_v3/person.jpg"
    img_org = Image.open(imgfile).convert("RGB")
    img_resized = img_org.resize((sz, sz))
    img_tensor = utils.image2torch(img_resized)

    return model, [img_tensor], {"pcc": pcc}


def test_yolov3_holli_pytorch(test_device):
    model, inputs, other = generate_model_yoloV3_imgcls_holli_pytorch(
        test_device,
        None,
    )

    compiled_model = forge.compile(model, sample_inputs=[inputs[0]])

def test_yolov3_holli_pytorch_1x1(test_device):
    # if test_device.arch == BackendDevice.Grayskull:
    #     pytest.skip()

    os.environ["FORGE_OVERRIDE_DEVICE_YAML"] = "wormhole_b0_1x1.yaml"
    os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"
    os.environ["FORGE_RIBBON2"] = "1"
    model, inputs, other = generate_model_yoloV3_imgcls_holli_pytorch(
        test_device,
        None,
    )
    compiled_model = forge.compile(model, sample_inputs=[inputs[0]])

