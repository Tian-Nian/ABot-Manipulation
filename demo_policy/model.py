import numpy as np
from XPolicyLab.model_template import ModelTemplate

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        # Initialize your policy model here according to model_cfg
    
    def set_language(self, instruction):
        # Process the instruction if needed
        print("[Model] Received instruction:", instruction)
        pass
    
    def update_obs(self, obs):
        # Update your model's observation here if needed
        print("[Model] Received observation:")
        pass

    def get_action(self):
        # Generate action according to your model and current observation
        # This is a dummy action for demonstration, replace it with your model's action
        if self.action_type == 'joint':
            action_dict = {
                "left_arm_joint_state": np.array([0] * 7),
                "left_ee_joint_state": np.array([0] * 1),
                "right_arm_joint_state": np.array([0] * 7),
                "right_ee_joint_state": np.array([0] * 1),
            }
        elif self.action_type == 'ee':
            action_dict = {
                "left_ee_pose": np.array([0] * 7),
                "left_ee_joint_state": np.array([0] * 1),
                "right_ee_pose": np.array([0] * 7),
                "right_ee_joint_state": np.array([0] * 1),
            }
        print("[Model] Generated action")
        return action_dict

    def reset(self):
        # Reset your model's internal state if needed
        print("[Model] Model successfully reset")
        pass