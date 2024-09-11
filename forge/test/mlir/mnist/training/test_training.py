# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

import forge
from ..utils import *


from loguru import logger

def test_mnist_training():
    torch.manual_seed(0)

    # Config
    num_epochs = 10
    batch_size = 1
    learning_rate = 0.001
    
    # Load dataset
    test_loader, train_loader = load_dataset(batch_size)

    # Load TensorBoard writer (for logging)
    writer = load_tb_writer()
    
    # Define model and instruct it to compile and run on TT device
    framework_model = MNISTLinear()

    # Create a torch loss and leave on CPU
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define optimizer and instruct it to compile and run on TT device
    framework_optimizer = torch.optim.SGD(framework_model.parameters(), lr=learning_rate)
    tt_model = forge.compile(framework_model, sample_inputs=[torch.rand(batch_size, 784)], loss=loss_fn, optimizer=framework_optimizer)

    logger.disable("")
    for epoch_idx in range(num_epochs):
        # Reset gradients (every batch)
        framework_optimizer.zero_grad()

        for name, param in framework_model.named_parameters():
            print(f"{name} = {param}")
            print(f"{name}.grad = {param.grad}")


        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            # Create target tensor and leave on CPU
            target = nn.functional.one_hot(target, num_classes=10).float()

            # Forward pass (prediction) on device
            pred = tt_model(data)[0]
            golden_pred = framework_model(data)

            # Compute loss on CPU
            loss = loss_fn(pred, target)

            golden_loss = loss_fn(golden_pred, target)

            total_loss += loss.item()
            
            # RUn backward pass on device
            loss.backward()
            
            grad = tt_model.backward(pred.grad)

            if batch_idx == 1000:
                break

            # Log gradients
            # for name, param in tt_model.named_parameters():
            #     writer.add_histogram(f"{name}.grad", param.grad, batch_idx)
            #
            # # Log loss
            # writer.add_scalar("Loss", loss.item(), batch_idx)

        # Adjust weights (on device)
        print(f"epoch: {epoch_idx} loss: {total_loss}")
        framework_optimizer.step()

    for batch_idx, (data, target) in enumerate(test_loader):
        pred = tt_model(data)[0]
        target = nn.functional.one_hot(target, num_classes=10).float()

        print(pred)
        print(target)

        print(loss_fn(pred, target))

        if batch_idx == 10:
            break
