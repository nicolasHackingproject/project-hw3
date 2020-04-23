import torch
import torch.autograd as autograd
import torch.nn as nn
import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import sys
import time
from env_ac import construct_task2_env

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))

model_name = 'adqn2_random_dR'
model_path = os.path.join(script_path, '{}.pt'.format(model_name))

learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
max_episodes  = 5000
t_max         = 300
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        self.current_number = 0
        self.buffer_limit = buffer_limit
        #self.buffer = deque() -> could be faster 
        self.buffer = []

        pass
    
    def push(self, transition):
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
        batch = []

        if self.current_number < batch_size:
            batch = random.sample(self.buffer, self.current_number)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        states_batch, actions_batch, rewards_batch, dones_batch, next_states_batch = list(map(torch.tensor, list(zip(*batch))))
        return states_batch, actions_batch, rewards_batch, dones_batch, next_states_batch


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
        # x = x.to(device)
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
    # resize tensors
    actions = actions.view(actions.size(0), 1)
    dones = dones.view(dones.size(0), 1)

    # compute loss
    # print(next_states.shape)
    next_Q = target.forward(next_states.float().to(device))
    curr_Q = model.forward(states.float().to(device)).gather(1, actions.to(device))
    
    max_next_Q = torch.max(next_Q, 1)[0]
    max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)

    expected_Q = rewards.to(device) + (1 - dones.int().to(device)) * gamma * max_next_Q
 
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

def train(model_class, env, pretrain=False, model= None, savepath=''):
    # Initialize model and target network
    if not(pretrain):
      model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    config = np.array([[5, 10, 0, 2], 
                       [5, 15, 2, 4],
                       [10, 20, 0, 3],
                       [10, 25, 3, 6],
                       [20, 30, 0, 3],
                       [20, 30, 3, 6],
                       [20, 35, 6, 9],
                       [30, 40, 0, 3],
                       [30, 40, 3, 6],
                       [30, 40, 6, 9],
                       [40, 49, 0, 3],
                       [40, 49, 3, 6],
                       [40, 49, 6, 9],
                       [49, 49, 9, 9]])
    len_config = len(config)

    for init in range(0, len_config):

        x = random.randint(config[init, 0], config[init, 1])
        y = random.randint(config[init, 2], config[init, 3])
        print('\nAgent\'s init position : x{}, y{}'.format(x, y))
        env = construct_task2_env(x, y, dense=True)

        # Initialize rewards, losses, and optimizer
        rewards = []
        losses = []
        lengths = []

        memory = ReplayBuffer()

        for episode in range(max_episodes):

            if episode % 200 == 0  and episode > 0:
                print('Test with several init and with dense rewards')
                mainTest(model, runs=50)
            elif episode == max_episodes-1 and init == len_config-1:
                print('Test with several init and with dense rewards')
                mainTest(model, runs=300)

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
                    lengths.append(t) 
                    break

            rewards.append(episode_rewards)

            # Train the model if memory is sufficient
            if len(memory) > batch_size*4:
                #if np.mean(rewards[print_interval:]) < 0.1:
                #    print('Bad initialization. Please restart the training.')
                #    exit()
                for i in range(train_steps):
                    loss = optimize(model, target, memory, optimizer)
                    losses.append(loss.item())

            # Update target network every once in a while
            if episode % target_update == 0:
                target.load_state_dict(model.state_dict())

            if episode % print_interval == 0 and episode > 0:
                # print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tepsilon : {:.1f}%".format(
                #             episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval*train_steps:]), epsilon*100))
                print("[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.4f},\tavg_length : {:.1f}".format(
                            episode, np.mean(rewards[print_interval:]), np.mean(losses[-print_interval:]), np.mean(lengths[-print_interval:])))

    return model

def get_model(modelpath=model_path):
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(modelpath)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model

def save_model(model, model_name):
    '''
    Save `model` to disk. Location is specified in `model_path`. 
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)

def get_env(i,j):
    '''
    Get the sample test cases for training and testing.
    '''
    return construct_task2_env(i,j, dense=False)



def mainTest(agent, runs=25):

    import sys
    import time
    from env import construct_task2_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        for run in range(runs):
            state = env.reset()
            agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
            #agent.initialize(**agent_init)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.act(state)   
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                    'done': done, 'info': info
                }
                #agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards)/len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(agent, task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            #agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            # if avg_rewards > 6:
            #     callback_name = model_name + '_timestep'
            #     save_model(agent, callback_name)
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task(runs=300):
        tcs = [('task_2_tmax50', 50), ('task_2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': runs, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task(runs=runs)
    timed_test(agent, task)


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    parser.add_argument('--test', dest='test', action='store_true', help='train the agent')
    parser.add_argument('--pretrain', dest='pretrain', action='store_true', help='train the agent')
    parser.add_argument('--path', dest='path', help='train the agent')
    parser.add_argument('--savepath', dest='savepath', help='train the agent')
    args = parser.parse_args()

    env = get_env(49,9)

    if args.train:
      print("train mode")
      model = train(AtariDQN, env)
      save_model(model, model_path)

    elif args.pretrain:
      print("pretrain mode")
      path_model = args.path
      savepath = args.savepath
      model = get_model(path_model)
      model_p = train(ConvDQN,env, pretrain=True,model= model,savepath=savepath)

    elif args.test:
      print("Test mode")
      FAST_DOWNWARD_PATH = "/fast_downward/"
      path_model = args.path
      model = get_model(path_model)
      mainTest(model, runs=300)