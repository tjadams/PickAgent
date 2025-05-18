# PickAgent: OpenVLA-powered Pick and Place Agent (Simulation)
OpenVLA is a open-source Vision-Language-Action (VLA) model with 7 billion parameters. Designed to empower robots with human-like perception and decision-making, it seamlessly integrates visual inputs and natural language instructions to perform diverse manipulation tasks. Trained on nearly a million episodes from the Open X-Embodiment dataset, OpenVLA sets a new standard for generalist robotic control. With a robust architecture combining SigLIP, DINOv2, and Llama 2 7B, it offers unparalleled adaptability and can be fine-tuned efficiently on consumer-grade GPUs, making advanced robotics more accessible than ever. [Project Page](https://openvla.github.io/)

## 🎬 1. Demo 
### 🌐 Gradio App Screenshot
<p align="center">
  <img src="https://github.com/user-attachments/assets/96928b5c-f683-4c75-b3ed-72b0f18dce92" alt="Screenshot" style="border: 10px solid black; border-radius: 5px;"/>
</p>

#### Watch in Youtube!
<p align="left">
<a href="https://www.youtube.com/watch?v=MvPEy6JLu94" title="PickAgent: OpenVLA-powered Pick and Place Agent(Simulation)"><img src="https://github.com/user-attachments/assets/2ee55647-412e-4522-b56b-2d4db4026c74" alt="PickAgent: OpenVLA-powered Pick and Place Agent(Simulation)" width="300px" align="left" /></a>
<a href="https://youtu.be/watch?v=MvPEy6JLu94" title="PickAgent: OpenVLA-powered Pick and Place Agent(Simulation)"><strong>PickAgent: OpenVLA-powered Pick and Place Agent(Simulation)</strong></a>
<div><strong>Gradio Demo</strong></div>
<br/>🚀 PickAgent is an AI-driven pick-and-place system powered by OpenVLA, showcasing advanced vision-language models in action. This simulation demonstrates how LLMs and computer vision work together for precise object manipulation.</p>

<br/>

### 🎥 Video Results
</div>
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%"><b></b></td>
        <td width="50%"><b></b></td>
  </tr>
  <tr>
    <td>
      <p>Prompt: Pick up the salad dressing and place it in the basket</p>
      <video src=https://github.com/user-attachments/assets/0672d86a-1397-45c8-bc89-9d7577fc1156 controls preload></video>    
    </td>
    <td>
       <p>Prompt: Pick up the tomato sauce and place it in the baske.</p>
      <video src=https://github.com/user-attachments/assets/56dd3c51-f049-4430-b523-da94d571a6f1 controls preload></video>    
    </td>
  </tr>
   <tr>
    <td>
      <p>Prompt: pick up the cream cheese and place it in the baske</p>
      <video src=https://github.com/user-attachments/assets/e2bad8d3-ab93-44c2-80bd-1ebbb7db186d controls preload></video>    
    </td>
    <td>
       <p>Prompt: Pick up the alphabet soup and place it in the bask.</p>
      <video src=https://github.com/user-attachments/assets/aa1ba68b-c8e1-446b-b65e-2724ff72ed2f controls preload></video>    
    </td>
   
  </tr>

</table>


## 🔧 2. Installation
NOTE: Some of the dependencies, simpler-env + ManiSkill2_real2sim + SAPIEN, are not available for Mac OS and also require a nVIDA GPU. Work in progress to migrate this to mujoco

```bash
# Create and activate conda environment
conda create -n pickagent python=3.11 -y
conda activate pickagent

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio # pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

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

pip install gradio mbodied mediapy transformers tensorflow timm
```

## 🚀 3. Inference
### Gradio App
```bash
conda activate pickagent
```

Run the Gradio app for inference:
```python
   python3 gradio_demo.py
```
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


### Command Line Interface
Run the python script for inference:
```python
   python3 inference.py --prompt="pick up the salad dressing and place it in the basket" --task="libero_object" --task_id=2 --image_resize=1024 --output_video="outputs/videos"
```

## 4. OpenVLA Models

<div align="center">

|      **Model**      | **Download**                                                                   |
| :-----------------: | ------------------------------------------------------------------------------ |
| General OpenVLA | [🤗 HuggingFace](https://huggingface.co/openvla/openvla-7b)  |
| OpenVLA - Finetuned Libero Spatial | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM-Qwen-0.5B) |
| OpenVLA - Finetuned Libero Object | [🤗 HuggingFace](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object) |
| OpenVLA - Finetuned Libero Goal | [🤗 HuggingFace](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal) |
| OpenVLA - Finetuned Libero 10 | [🤗 HuggingFace](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10) |
</div>

## 🙏 5. Acknowledgement
models are borrowed from [OpenVLA](https://openvla.github.io/)

