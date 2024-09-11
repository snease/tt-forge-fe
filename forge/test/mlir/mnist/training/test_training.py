# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

import forge
from ..utils import *

def test_mnist_training():
    torch.manual_seed(0)

    # Config
    num_epochs = 4
    batch_size = 1
    learning_rate = 0.1
    
    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer (for logging)
    writer = load_tb_writer()
    
    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear()

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.L1Loss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    tt_model = forge.compile(framework_model, sample_inputs=[torch.rand(1, 784)], loss=loss_fn, optimizer=framework_optimizer)

    for epoch_idx in range(num_epochs):
        # Reset gradients (every batch)
        framework_optimizer.zero_grad()

        for name, param in framework_model.named_parameters():
            print(f"{name} = {param}")
            print(f"{name}.grad = {param.grad}")

        for batch_idx, (data, target) in enumerate(train_loader):
            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            pred = tt_model(data)[0]
            golden_pred = framework_model(data)
            
            print(pred)
            print(golden_pred)
            print(target)
            # Compute loss on CPU
            loss = loss_fn(pred, target)
            print(f"loss = {loss}")

            golden_loss = loss_fn(golden_pred, target)
            print(f"golden_loss = {golden_loss}")
            
            # RUn backward pass on device
            loss.backward()
            
            grad = tt_model.backward(pred.grad)

            return

            # Log gradients
            # for name, param in tt_model.named_parameters():
            #     writer.add_histogram(f"{name}.grad", param.grad, batch_idx)
            #
            # # Log loss
            # writer.add_scalar("Loss", loss.item(), batch_idx)

        # Adjust weights (on device)
        framework_optimizer.step()

        for param in framework_model.parameters():
            print(f"{param} , grad = {param.grad}")

