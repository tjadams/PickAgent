# PickAgent: OpenVLA-powered Pick and Place Agent (Simulation)
OpenVLA is a open-source Vision-Language-Action (VLA) model with 7 billion parameters. Designed to empower robots with human-like perception and decision-making, it seamlessly integrates visual inputs and natural language instructions to perform diverse manipulation tasks. Trained on nearly a million episodes from the Open X-Embodiment dataset, OpenVLA sets a new standard for generalist robotic control. With a robust architecture combining SigLIP, DINOv2, and Llama 2 7B, it offers unparalleled adaptability and can be fine-tuned efficiently on consumer-grade GPUs, making advanced robotics more accessible than ever. [Project Page](https://openvla.github.io/)

### Demo 
### Gradio App
![Screenshot from 2025-03-24 15-46-29](https://github.com/user-attachments/assets/e4b6a646-e209-40c0-bd3d-1a028d92cde7)


### Video results


### Usage
#### Installation

```bash
# Create and activate conda environment
conda create -n openvla python=3.11 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
Additionally, install other required packages for simulation:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

### Gradio Demo
Here’s a summary of the Gradio inputs and outputs
Inputs:
 - Task: Selects the task type for the simulation.
 - Task ID: Specifies the task instance ID.
 - Prompt: Input for natural language instructions to control the robot.
 - Preview Button: Updates the environment preview based on selected task.
 - Run Simulation Button: Run the simulation with the given prompt.

Outputs:
 - Preview Image: Shows the environment's first frame.
 - Simulation Video: Shows the simulation result video.


### OpenVLA Models

<div align="center">

|      **Model**      | **Download**                                                                   |
| :-----------------: | ------------------------------------------------------------------------------ |
| General OpenVLA | [🤗 HuggingFace](https://huggingface.co/openvla/openvla-7b)  |
| OpenVLA - Finetuned Libero Spatial | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM-Qwen-0.5B) |
| OpenVLA - Finetuned Libero Object | [🤗 HuggingFace](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object) |
| OpenVLA - Finetuned Libero Goal | [🤗 HuggingFace](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal) |
| OpenVLA - Finetuned Libero 10 | [🤗 HuggingFace](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10) |

