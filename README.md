# Reinforcement Learning for Flexible back Quadruped

[![YouTube](http://i.ytimg.com/vi/1mVXlgvw5_c/hqdefault.jpg)](https://www.youtube.com/watch?v=jfLxBojE-_s)

### Overview
This repository provides the necessary code to run simulations in the MuJoCo environment, complete with step-by-step instructions for downloading and configuring MuJoCo seamlessly. It includes custom environments designed for simulating:

A full 8-DOF quadruped capable of walking.
A unique segmented-back quadruped structure for enhanced locomotion.
These simulations are intended to bridge the gap between virtual models and real-world deployment, particularly for a physical quadruped robot that I have personally designed and manufactured.

Additionally, the repository contains simulation code for the Stanford Pupper robot, adapted with a segmented-back design.

For the detailed explanation please visit my [portfolio](https://sdalal1.github.io/projects/Quadruped-design-and-Improved-gaits/) post.


### Repository Structure
The repository is organized into the following folders:

- old_approaches - This folder contains earlier versions of the code and setups that were implemented before transitioning to reinforcement learning (RL). While some files may be incomplete, they serve as valuable references for previous approaches and methodologies.

- RL_final_project - This folder documents various iterations of training and testing code for MuJoCo simulations. Each version corresponds to specific MuJoCo models, highlighting the progression and development of reinforcement learning strategies.

- muj_models - This folder houses all the MuJoCo models created during the project. These models provide detailed insights into the sequential modifications made to optimize the quadruped’s structure and performance.

- motor_code - Some code setup to test motors with RS485 CAN-hat and raspberry pi4B.

This repository aims to support advancements in efficient quadruped locomotion by combining simulation and real-world implementations.

### To run everything create a Python virtual env
```bash
sudo apt install -y python3-venv wget
python -m venv .venv
source .venv/bin/activate
```

### Download Mujoco 2.1.0
```bash
wget -O /tmp/mujoco210-linux-x86_64.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p $HOME/.mujoco/mujoco210
rm -rf $HOME/.mujoco/mujoco210/*
tar -xvzf /tmp/mujoco210-linux-x86_64.tar.gz --directory $HOME/.mujoco/mujoco210 --verbose
```
### Download other dependencies
```bash
cat requirements.txt | xargs -n 1 python -m pip install
```
### Training a Model

The files on the homepage of this repository represent the finalized versions used for training and testing. These scripts are optimized with multiprocessing to enable faster training.

To initiate training, use the following command:
```bash
python full_model_script_multi_processs.py FullCheetah-v0 PPO dummy --train --fname *model_name*
```

To initiate subprocvec training:
``` bash
python test_multi.py
```
All the required things are setup inside this script and can be changes as per user requirements. These changes will include model_folder, logs_folder, number of parallel environments.

- The training script automatically saves the trained models to the specified directory. This directory can be adjusted directly in the script.

- A dedicated logging folder captures all training data, which can be visualized using TensorBoard for real-time monitoring of the training process.

To visualize the trained model
```bash
python full_model_script_multi_processs.py FullCheetah-v0 PPO subproc --test *model_name*
```

A pre-trained model is included in this repository for demonstration purposes. You can visualize its behavior with the following command:
``` bash
python full_model_script_multi_processs.py FullCheetah-v0 PPO subproc --test RL_final_project/working_3D_subproc_1_link.PPO
```

These scripts simplify the training pipeline and ensure seamless testing of reinforcement learning models.

