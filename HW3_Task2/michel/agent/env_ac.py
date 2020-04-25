import gym
from gym_grid_driving.envs.grid_driving import LaneSpec, Point, DenseReward, SparseReward

def construct_task2_env(i,j, dense=False):
    if dense:
        rewards = DenseReward
    else:
        rewards = SparseReward
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50, 
              'agent_pos_init' : Point(i,j), 'rewards': rewards,
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -1]), 
                        LaneSpec(cars=6, speed_range=[-1, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=7, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -2]), 
                        LaneSpec(cars=7, speed_range=[-1, -1]), 
                        LaneSpec(cars=6, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -2])]
            }
    return gym.make('GridDriving-v0', **config)