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
    