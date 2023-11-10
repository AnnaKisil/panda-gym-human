import gym
import pybullet as p
import time 
import panda_gym

import panda_gym_human

from panda_gym_human.envs.panda_tasks.pick_and_place_human_env import PickAndPlaceHumanEnv



# visualise environment, random robot movements
if __name__ == "__main__":
    env = PickAndPlaceHumanEnv()
    observation = env.reset()

    for i in range(1000):
       # time.sleep(5)
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)

        if done:
            obs = env.reset()

# visualise trained model
# def main():
#     parser = argparse.ArgumentParser(description="Load and visualize a trained RL model")
#     parser.add_argument("--env_id", type=str, required=True, help="Environment ID")
#     parser.add_argument("--algorithm_name", type=str, required=True, help="Algorithm name")
#     args = parser.parse_args()

#     # Create the environment
#     env = gym.make(args.env_id + '-v1', render=True)

#     # Define the model path based on convention (without algorithm name as a suffix)
#     model_path = f"./trained/{args.env_id}/{args.env_id}_{args.algorithm_name}"

#     # Load the trained model
#     if args.algorithm_name == 'TRPO':
#         model = PPO.load(model_path, env=env)
#     elif args.algorithm_name == 'SAC':
#         model = SAC.load(model_path, env=env)
#     elif args.algorithm_name == 'TQC':
#         model = TQC.load(model_path, env=env)
#     else:
#         raise ValueError("Unsupported algorithm name")

#     obs = env.reset()
#     for i in range(1000):
#         action, _state = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         print(info)
#         env.render()
#         if done:
#             print('Done')
#             obs = env.reset()

# if __name__ == "__main__":
#     main()

# '''
#         'Capping', 

#         'EmptyRackInsertion',
#         'LoadedRackInsertion',

#         'VialInsertion_From_SingleHolder_to_SingleHolder',
#         'VialInsertion_From_SingleHolder_to_Rack',
#         'VialInsertion_From_SingleHolder_to_LoadedRack',

#         'VialInsertion_From_Rack_to_Rack',
#         'VialInsertion_From_LoadedRack_to_Rack',
#         'VialInsertion_From_Rack_to_LoadedRack',
# '''

# # Example
# # python show_main.py --env_id 'LoadedRackInsertion' --algorithm_name 'TQC'
