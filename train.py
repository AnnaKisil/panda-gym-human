import argparse
import gym
import panda_gym
from stable_baselines3 import SAC, PPO, DDPG, HerReplayBuffer, TD3
# from sb3_contrib import TQC, TRPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import panda_gym_human

def main(args):
    env_id = args.environment
    log_dir = './tensorboard/' + env_id
    total_timesteps = args.timesteps

    eval_env = gym.make('PickAndPlaceHuman-v1', render=True)
    # eval_env = gym.make('custom_envs/' + env_id + '-v1', render=True)

    # eval_callback = EvalCallback(eval_env, best_model_save_path="./best_models_path/" + env_id,
    #                               log_path="./log_path/" + env_id, eval_freq=100_000,
    #                               deterministic=True, render=False)

    algorithms = {
        'SAC': SAC,
        # 'TQC': TQC,
        'TD3': TD3,
        'DDPG': DDPG
    }

    algorithm_cls = algorithms[args.algorithm]
    
    model = algorithm_cls(policy="MultiInputPolicy", env=eval_env, learning_rate=1e-3,
                          buffer_size=2_000_000,
                            batch_size=2048,
                         replay_buffer_class=HerReplayBuffer, seed=args.seed,
                          policy_kwargs=dict(net_arch=[1024, 1024, 1024, 1024]),
                          replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
                          gamma=0.95, tau=0.05,
                            verbose=1, tensorboard_log=log_dir)

    model.learn(total_timesteps=total_timesteps)

    model.save('./trained/' + env_id + '/' + env_id + '_' + model.__class__.__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL models on different environments")
    parser.add_argument("--algorithm", type=str, choices=['SAC', 'TD3', 'DDPG'],
                        default='SAC', help="RL algorithm to use")
    parser.add_argument("--environment", type=str, choices=['PickAndPlaceHuman'], default='PickAndPlaceHuman', help="Environment ID")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total timesteps for training")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args, _ = parser.parse_known_args()

    main(args)

    # Example
    # python train_main.py --algorithm 'SAC' --timesteps 2_000_000 --environment 'PickAndPlaceHuman' --seed 42
