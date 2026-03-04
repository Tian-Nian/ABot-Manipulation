import argparse
from client_server.model_client import ModelClient

class TestEnv:
    def __init__(self, deploy_cfg):
        self.success_num, self.episode_num = 0, 0
        self.deploy_cfg = deploy_cfg
        self.episode_step_limit = 5
        
        self.model_client = ModelClient(port=deploy_cfg['port'])

    def get_obs(self):
        # v1.0
        demo_obs = {
            
        }
        return demo_obs

    def eval_one_episode(self):
        policy_name = self.deploy_cfg['policy_name']
        try:
            eval_module = __import__(f'XPolicyLab.{policy_name}.deploy', fromlist=['eval_one_episode'])
        except ImportError as e:
            print("[TestEnv]", f"Failed to import policy module: XPolicyLab.{policy_name}.deploy. Error: {e}", "ERROR")
            raise e
            
        if not hasattr(eval_module, 'eval_one_episode'):
            print("[TestEnv]", f"Module '.{policy_name}.deploy' does not have 'eval_one_episode' function", "ERROR")
            raise AttributeError(f"Missing eval_one_episode in policy module")
            
        eval_module.eval_one_episode(TASK_ENV=self, model_client=self.model_client)

    def reset(self):
        self.model_client.call(func_name="reset")
        self.episode_step = 0

    def get_instruction(self):
        instruction = "Language instruction for the task"  # Replace with actual instruction retrieval logic
        print("[TestEnv] Get Instruction:", instruction)
        return instruction

    def take_action(self, action):
        print(f"[TestEnv] Action Step: {self.episode_step} / {self.episode_step_limit} (step_limit)", end='\r')
        self.episode_step += 1
        # check action validity here if needed

    def is_episode_end(self):
        print("[TestEnv] Check Episode End:", self.episode_step >= self.episode_step_limit)
        return self.episode_step >= self.episode_step_limit
    
    def finish_episode(self):
        print("[TestEnv] Episode finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--base_cfg", type=str, required=True)
    parser.add_argument("--policy_name", type=str, required=True, help="XPolicyLab module name for deployment")
    parser.add_argument("--port", type=int, required=True, help="server port")
    parser.add_argument("--eval_episode_num", type=int, default=100, help="number of evaluation episodes")

    args_cli = parser.parse_args()
    deploy_cfg = vars(args_cli)          # 或 args_cli.__dict__.copy()
    test_env = TestEnv(deploy_cfg)

    # Load XPolicyLab
    for idx in range(10):
        print(f"\033[94m🚀 Running Episode {idx}\033[0m")
        test_env.reset() # reset model, robot, and environment
        test_env.eval_one_episode()
        test_env.finish_episode()