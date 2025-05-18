import argparse
from openvla import OpenVLA
from PIL import Image
import numpy as np
import os
#import simpler_env
import mediapy
#import sapien.core as sapien
from mbodied.robots import Robot
#from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from robot_utils import (
    get_libero_env,
    GenerateConfig,
    get_image_resize_size,
    get_task,
    get_libero_dummy_action,
    get_libero_image,
    normalize_gripper_action,
    invert_gripper_action,
    save_rollout_video,
    quat2axisangle
)

class PickAgent:
    def __init__(self, task="libero_object", task_id=0, image_resize=1024, output_video=None):
        """
        Initialize the OpenVLA agent with configuration parameters.
        """
        self.cfg = GenerateConfig()
        self.cfg.task_suite_name = task
        self.cfg.num_steps_wait = 20
        self.output_video = output_video
        self.task_name = task
        self.task_id = task_id
        self.custom_resize = image_resize
        self.cfg.unnorm_key = self.cfg.task_suite_name

        # Initialize OpenVLA
        model_name = self.get_model_name(task)
        self.openvla = OpenVLA(model_name=model_name)
        
        # Load task and environment
        self.setup_environment()
    
    def get_model_name(self, task):
        model_name = "openvla/openvla-7b"
        if task=="libero_object":
            model_name = "openvla/openvla-7b-finetuned-libero-object"
        elif task=="libero_spatial":
            model_name = "openvla/openvla-7b-finetuned-libero-spatial"
        elif task=="libero_goal":
            model_name = "openvla/openvla-7b-finetuned-libero-goal"
        elif task=="libero_10":
            model_name = "openvla/openvla-7b-finetuned-libero-10"
        elif task=="general":
            model_name = "openvla/openvla-7b"
        return model_name

    def setup_environment(self):
        """Set up the task and environment."""

        if self.task_name == "libero_spatial":
            self.max_steps = 220
        elif self.task_name == "libero_object":
            self.max_steps = 280
        elif self.task_name == "libero_goal":
            self.max_steps = 300
        elif self.task_name == "libero_10":
            self.max_steps = 520
        elif self.task_name == "libero_90":
            self.max_steps = 400
        else:
            self.max_steps = 280

        self.task, self.task_suite = get_task(self.cfg, self.task_id)
        self.initial_states = self.task_suite.get_task_init_states(self.task_id)
        self.env, task_description = get_libero_env(self.task, self.cfg.model_family, resolution=256)

        if self.custom_resize:
            self.resize_size = self.custom_resize
        else:
            self.resize_size = get_image_resize_size(self.cfg)
        
        # Set unnormalization key
        if hasattr(self.openvla, 'model') and hasattr(self.openvla.model, 'norm_stats'):
            if self.cfg.unnorm_key not in self.openvla.model.norm_stats and f"{self.cfg.unnorm_key}_no_noops" in self.openvla.model.norm_stats:
                self.cfg.unnorm_key = f"{self.cfg.unnorm_key}_no_noops"
    
    def process_observation(self, obs):
        """
        Process raw observation into model input format.
        """
        # Get preprocessed image
        img = get_libero_image(obs, self.resize_size)
        
        # Prepare observations dict with state information
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (
                    obs["robot0_eef_pos"], 
                    quat2axisangle(obs["robot0_eef_quat"]), 
                    obs["robot0_gripper_qpos"]
                )
            ),
        }
        
        return observation, img
    
    def run_simulation(self, prompt, episode_idx=0):
        """
        Run a full simulation episode with the given prompt.
        """
        # Reset environment and set initial state
        self.env.reset()
        obs = self.env.set_init_state(self.initial_states[episode_idx])
        
        t = 0
        replay_images = []
        success = False
        
        print(f"Starting simulation with prompt: '{prompt}'")
        
        while t < self.max_steps + self.cfg.num_steps_wait:
            # Wait period with dummy actions
            if t < self.cfg.num_steps_wait:
                obs, reward, done, info = self.env.step(get_libero_dummy_action(self.cfg.model_family))
                t += 1
                continue
            
            # Process observation
            observation, img = self.process_observation(obs)
            print("img: ", img.shape, type(img))
            replay_images.append(img)
            
            # Get action from OpenVLA model
            action = self.openvla.get_action(
                observation=observation,
                prompt=prompt,
                unnorm_key=self.cfg.unnorm_key
            )
            
            # Log action for debugging
            print(f"Action = {action}")
            
            # Normalize and apply action
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, reward, done, info = self.env.step(action.tolist())
            
            t += 1
            
            # Check for episode completion
            if done:
                success = True
                print(f"Episode completed successfully in {t} steps!")
                break
                
        if t >= self.max_steps + self.cfg.num_steps_wait:
            print("Episode reached maximum steps without completion.")
        
        # Save replay video
        video_path = save_rollout_video(
            replay_images, 
            self.task_id, 
            success=success, 
            task_description=prompt,
            output_dir=self.output_video
        )
        print(f"Saved replay video to {video_path}")
        
        return success, replay_images
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenVLA Inference (PickAgent)")
    
    parser.add_argument("--prompt", type=str, required=True,
                        help="instruction for the robot")
    parser.add_argument("--task", type=str, default="libero_object",
                        help="Task name, choose from this list [libero_object, libero_spatial, libero_goal, libero_10, general](default: libero_object)")
    parser.add_argument("--task_id", type=int, default=0,
                        help="task_id can be selected from [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  (default: 0)")
    parser.add_argument("--image_resize", type=int, default=1024,
                        help="Image resize dimensions, format: width or height (default: 1024)")
    parser.add_argument("--output_video", type=str, default="outputs/videos",
                        help="Output directory for videos (default: outputs/videos)")
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    # Initialize the pick agent
    agent = PickAgent(
        task=args.task,
        task_id=args.task_id,
        image_resize=args.image_resize,
        output_video=args.output_video,
    )
    # Run single simulation
    success, _ = agent.run_simulation(args.prompt)
    print(f"\nSimulation completed with {'success' if success else 'failure'}")

