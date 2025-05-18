from transformers import AutoModelForVision2Seq, AutoProcessor
from mbodied.types.motion.control import HandControl
from robot_utils import crop_and_resize
from PIL import Image
import torch
import os
import tensorflow as tf
import numpy as np

class OpenVLA:
    # "cuda:0" device for nvidia, mps for mac, cpu for ubuntu vm on mac
    def __init__(self, model_name="openvla/openvla-7b-finetuned-libero-object", device="mps"):
        """
        Initialize the Open VLA model  
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            #attn_implementation="flash_attention_2",
            #torch.bfloat16 not supported on mps
            #torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        print("Model loaded successfully!")
    
    def get_action(self, observation, prompt, unnorm_key, center_crop=False):
        """Generates an action with the VLA policy."""
        image = Image.fromarray(observation["full_image"])
        image = image.convert("RGB")

        if center_crop:
            batch_size = 1
            crop_scale = 0.9

            # Convert to TF Tensor and record original data type (should be tf.uint8)
            image = tf.convert_to_tensor(np.array(image))
            orig_dtype = image.dtype

            # Convert to data type tf.float32 and values between [0,1]
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Crop and then resize back to original size
            image = crop_and_resize(image, crop_scale, batch_size)

            # Convert back to original data type
            image = tf.clip_by_value(image, 0, 1)
            image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

            # Convert back to PIL Image
            image = Image.fromarray(image.numpy())
            image = image.convert("RGB")

        # Build VLA prompt
        prompt = f"In: What action should the robot take to {prompt.lower()}?\nOut:"

        # Process inputs.
        # torch.bfloat16 not supported on mps
        #inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.float16)

        # Get action.
        action = self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        return action