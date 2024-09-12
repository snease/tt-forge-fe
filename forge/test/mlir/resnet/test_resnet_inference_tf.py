# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
# from tf.keras import applications
# import tf.keras.applications.ResNet50 as resnet50

import forge

def test_resnet_inference():
    # Compiler configurations
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False
    forge.verify.config._get_global_verify_config().verify_forge_codegen_vs_framework = True

    # Load ResNet50 model
    framework_model = tf.keras.applications.ResNet50()

    input_image = tf.random.uniform((1, 224, 224, 3), dtype=tf.bfloat16)

    # Sanity run
    generation_output = framework_model(input_image)
    print(generation_output)

    # Compile the model
    compiled_model = forge.compile(framework_model, input_image)
