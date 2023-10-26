# COCI:Convergence-aware Optimal Checkpointing for Exploratory Deep Learning Training Jobs

This source code is available under the [MIT License](LICENSE.txt).

## Introduction

COCI is an approach to compute optimal checkpointing configuration for a exploratory DL training job.  This implementation is built atop PyTorch.   The key idea of COCI is to choose convergence progress directly as a metric of checkpointing so that emphasizing the protections of early iterations, where a DL model gains more convergence progress, and more costly to recover in data parallelism.

## Installation
### Prerequisites

* Python3.8.+
* PyTorch-1.10.+
* numpy-1.24.+
* scipy-1.10.+
* json-2.0.+

### Quick Start

    git clone https://github.com/wangzc-HPC/COCI.git
    cd COCI
    pip install -r requirements.txt
    python ./models/ResNet/ResNet101_CIFAR10.py

## Using COCI
COCI is a pluggable module compared with PyTorch. It requires no extra user input.
    
 1. Import COCI iterator in the training script
  
          from src.COCI_Iterator import COCIIterator

 2. Initialize a checkpoint wrapper that tracks state to be checkpointed. 
  
          COCI = COCIIterator(model_name='resnet101',
                              dataloader=train_loader,
                              ft_lambda=0.0042,
                              epoch=NUM_EPOCHS,
                              model=model,
                              optimizer=optimizer)
* model_name: the name of training model, the value type is "string", required field
* dataloader: a dataloader for training dataset, usually defined at the beginning of the training code, required field
* ft_lambda: failure occurrence rate, required field
* epoch: the total epoch number, usually defined at the beginning of the training code, required field
* ck_mode: the mode of checkpointing. It's a string with value of "AUTO" or "MANUAL", default is "AUTO", optional field
* ts: the cost of taking a checkpoint, the unit is minute. If the "ck_mode" is "MANUAL", this parameter is a required field, otherwise, it's an optional field.
* theta_1: the slope of the fitting loss curve. If the "ck_mode" is "MANUAL", this parameter is a required field, otherwise, it's an optional field.
* theta_2: the deviation of the fitting loss curve. If the "ck_mode" is "MANUAL", this argument is a required field, otherwise, it's an optional field.
* fit_interval: the number of iterations at each fitting loss curve interval, optional field
* profile_threshold: the number of iterations for making a profile at the beginning of training, optional field
* user-defined checkpoint content, at least model and optimizer, such as "model=model", "optimizer=optimizer", "checkpoint_time=checkpoint_time". It's a required field.

 3. Replace the original `optimizer.step()` with `COCI.optimizer_step()`.
  
          COCI.optimizer_step(loss, model, optimizer)

 4. Use the interface `COCI.recovery()` provided by COCI to restore model, optimizer and dataloader.
  
          flag, start_epoch= COCI.recovery()
