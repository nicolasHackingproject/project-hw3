#### TEST FILE FOR REPLAY BUFFER

import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Import 
buffer_limit = 3

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.current_number = 0
        self.buffer_limit = buffer_limit
        #self.buffer = deque() -> could be faster 
        self.buffer = []

        pass
    
    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        '''

        """Add an experience to the buffer"""
    
        #Add transition tupple to buffer
        # If buffer is not full 
        if self.current_number < self.buffer_limit:
            self.buffer.append(transition)
            self.current_number += 1
        # If buffer is full -> pop one element 
        else:
            # Remove first element of the list 
            self.buffer.pop(0)
            self.buffer.append(transition)

        pass
    
    def sample(self, batch_size):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        batch = []

        if self.current_number < batch_size:
            batch = random.sample(self.buffer, self.current_number)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        
        states_batch, actions_batch, rewards_batch, dones_batch, next_states_batch = list(map(torch.tensor, list(zip(*batch))))
       

        return states_batch, actions_batch, rewards_batch, dones_batch, next_states_batch

        pass

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)


if __name__ == '__main__':
    
    buffer_instance = ReplayBuffer(12)
    print(buffer_instance.buffer_limit)
    print(buffer_instance.buffer)

    for i in range(14):
        #buffer_instance.push(("state{}".format(i), "action{}".format(i), i, "ns{}".format(i), "d{}".format(i)))
        buffer_instance.push((i, i, i, i, i))

    print(buffer_instance.buffer)
    
    resFinal = buffer_instance.sample(3)

    print(resFinal)
    


    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        for run in range(runs):
            state = env.reset()
            agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
            #agent.initialize(**agent_init)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)   
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards)/len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task():
        tcs = [('task_2_tmax50', 50), ('task_2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task()
    timed_test(task)