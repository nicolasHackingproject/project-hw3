import torch
import torch.autograd as autograd
import torch.nn as nn
import env as env_builder
import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import sys
import time
from env import construct_task2_env

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))

date_save = time.strftime("%d %H %M")
model_path = os.path.join(script_path, 'model_{}.pt'.format(date_save))

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
# buffer_limit = 5000
buffer_limit  = 10000
# batch_size = 32
batch_size    = 64

max_episodes  = 2000
t_max         = 600
# min_buffer = 2000
min_buffer    = 4000
target_update = 100 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
#psilon_decay = 500
epsilon_decay = 700
#print_interval= 20
print_interval= 10


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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



class Base(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)

class BaseAgent(Base):
    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''

        #Random value  
        rand_val = np.random.random()
        if rand_val < epsilon:
            #Return random action with outpout < num_actions        
            
            output = np.random.randint(0, self.num_actions)
        else:
            #Use forward prediction to choose action 
          
            output = torch.argmax(self.forward(state), dim=1)
        
        return int(output)
    

class DQN(BaseAgent):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

class ConvDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        super().construct()

class AtariDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )


def compute_loss(model, target, states, actions, rewards, next_states, dones):
    
    '''
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    '''

    '''compute_loss: compute the temporal difference loss  of the DQN model Q, incorporating
    the target network ^ Q:  = Q(s; a) 􀀀 (r + 
    maxa ^Q
    (s0; a); with discounting factor 
    and
    reward r of state s after taking action a and giving next state s0. To minimize this distance,
    we can use Mean Squared Error (MSE) loss or Huber loss. Huber loss is known to be less
    sensitive to outliers and in some cases prevents exploding gradients (e.g. see Fast R-CNN
    paper by Ross Girshick1).'''

    '''Target Network: DQN loss function computes the distance between the Q values induced
    by the model with its expected Q values (given by the next state Q values and current state
    reward). However, such an objective is hard to optimize because it optimizes for a moving
    target distribution (i.e., the expected Q values). To minimize such a problem, instead of
    computing the expected Q values using the model that we are currently training, we use a
    target model (target network) instead, a replica of the model that is synchronized every few
    episodes.'''

    # resize tensors
    actions = actions.view(actions.size(0), 1)
    dones = dones.view(dones.size(0), 1)

    # compute loss
    next_Q = target.forward(next_states.float())
    curr_Q = model.forward(states.float()).gather(1, actions)
    
    max_next_Q = torch.max(next_Q, 1)[0]
    max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
    #print(max_next_Q)
    expected_Q = rewards + (1 - dones.int()) * gamma * max_next_Q
    #print(expected_Q)
    loss = F.mse_loss(curr_Q, expected_Q.detach()) 
    return loss

def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def train(model_class, env):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).
    
    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the 
    `optimize` function.

    Output: `model`: the trained model.
    '''

    # Initialize model and target network
    model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    # Initialize replay buffer
    memory = ReplayBuffer()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses_l= []
    rewards_l= []
    episodes_l= []

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            action = model.act(state, epsilon)
            
            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
        
        # Train the model if memory is sufficient
        if len(memory) > min_buffer:
            if np.mean(rewards[print_interval:]) < 0.1:
                print('Bad initialization. Please restart the training.')
                exit()
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*10:]), len(memory), epsilon*100))
            episodes_l.append(episode)
            rewards_l.append(np.mean(rewards[print_interval:]))
            losses_l.append(np.mean(losses[print_interval*10:]))
    try:
      object_export = np.array([episodes,rwds,losses_l])
    except:
      print("Error while exporting")
      data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
      torch.save(data, model_path)
    
    return model    

def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model

def save_model(model):
    '''
    Save `model` to disk. Location is specified in `model_path`. 
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, 'history_{}'.format(date_save))

def get_env():
    '''
    Get the sample test cases for training and testing.
    '''

    return env_builder.construct_task2_env()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()

    env = get_env()

    if args.train:
        model = train(AtariDQN, env)
        save_model(model)
    else:
        FAST_DOWNWARD_PATH = "/fast_downward/"
        model = get_model()
    #test(model, env, max_episodes=600)


),
            nn.Linear(512, self.num_actions)
        )