# CS6910: Assignment 3

Authors: Siddharth Nishtala, Richa Verma

## Steps to run
Run the following commands:

    git clone https://github.com/siddharthnishtala/cs6910_assignment3

    cd cs6910_assignment3
    
    pip install -r requirements.txt

    wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar

    tar -xf dakshina_dataset_v1.0.tar
    
Sweeps can be setup using [train.py](https://github.com/siddharthnishtala/cs6910_assignment3/blob/master/train.py), [config_vanilla.yaml](https://github.com/siddharthnishtala/cs6910_assignment3/blob/master/config_vanilla.yaml) and [config_attention.yaml](https://github.com/siddharthnishtala/cs6910_assignment3/blob/master/config_vanilla.yaml). The commands to run the vanilla sweep are given below:

    wandb sweep config_vanilla.yaml

    wandb agent sweep-id    

The commands to run the attention sweep are given below:

    wandb sweep config_attention.yaml

    wandb agent sweep-id    