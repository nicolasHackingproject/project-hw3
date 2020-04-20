
import sys
import torch  
import gym
import numpy as np  
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from env import construct_task2_env
import os
import sys
import time
import random

# hyperparameters
hidden_size = 1096
learning_rate = 3e-3

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 5000
change_mod = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_path = os.path.dirname(os.path.realpath(__file__))
date_save = time.strftime("%d_%H_%M")



def save_model(model,model_path='',Actor="Actor"):
    '''
    Save `model` to disk. Location is specified in `model_path`. 
    '''
    #model_path_actor = os.path.join(script_path, 'ac_{}_{}.pt'.format("Actor",date_save))
    data = (model.__class__.__name__, model.state_dict())
    torch.save(data, model_path)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = torch.flatten(state, start_dim=1).to(device)
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

def a2c(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    actor_critic = ActorCritic(2000, env.action_space.n,hidden_size).to(device)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for inter in range(7,9):
      
      max_episodes = min(5000*(inter+1),20000)
      x_min = min(10+(inter+1)*5,49)
      #x_max = min(int(episode/500),49)
      y_min = 0
      y_max = 5
      
      i_1= random.randint(min(x_min,49),min(x_min+2,50))
      print(x_min,i_1)
      if x_min ==49:
        j_1 = 4
      else:
        j_1= random.randint(y_min,y_max)
    
      #env = construct_task2_env(i_1,j_1)
      #env = construct_task2_env(49,4)
      print("environment change: x_range debut {} x_range max debut {}".format(x_min, min(x_min+2,49)))
      state = env.reset()


      for episode in range(max_episodes):
          log_probs = []
          values = []
          rewards = []
          if episode % 1000 == 0:
            i_1= random.randint(min(x_min,49),min(x_min+2,50))
          if x_min == 49:
            j_1 = 4
          else:
            j_1= random.randint(y_min,y_max)
            print("environment change: i debut {} j debut {}".format(i_1, j_1))
          env = construct_task2_env(i_1,j_1)
          state = env.reset()

          for steps in range(num_steps):
              value, policy_dist = actor_critic.forward(state)
              value = value.detach().numpy()[0,0]
              dist = policy_dist.detach().numpy() 

              action = np.random.choice(num_outputs, p=np.squeeze(dist))
              log_prob = torch.log(policy_dist.squeeze(0)[action])
              entropy = -np.sum(np.mean(dist) * np.log(dist))
              new_state, reward, done, _ = env.step(action)
              #if not isinstance(state, torch.FloatTensor):
              #  if not isinstance(state, torch.FloatTensor):
              #    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
              #  state = torch.flatten(state, start_dim=1).to(device)
              #  print(state.size())

              if done and not(reward) == 10 and steps > 0:
                reward = -0.1*steps
              if done and reward == 10 and x_min == 49:
                if steps <= 40:
                  reward = 20 
                else:
                  reward = 10

              rewards.append(reward)
              values.append(value)
              log_probs.append(log_prob)
              entropy_term += entropy
              state = new_state
              

              if done or steps == num_steps-1:
                  
                  Qval, _ = actor_critic.forward(new_state)
                  Qval = Qval.detach().numpy()[0,0]
                  all_rewards.append(np.sum(rewards))
                  all_lengths.append(steps)
                  average_lengths.append(np.mean(all_lengths[-10:]))
                  if episode % 10 == 0:                    
                      try:
                        sys.stdout.write("episode: {}, nb rwd {},average reward: {}, cur inter: {}, average length: {} \n".format(episode, len(all_rewards[-10:]),np.mean(all_rewards[-10:]), inter, average_lengths[-1]))
                      except:
                        print("non initialize")
                  break
          
          # compute Q values
          Qvals = np.zeros_like(values)
          for t in reversed(range(len(rewards))):
              Qval = rewards[t] + GAMMA * Qval
              Qvals[t] = Qval
    
          #update actor critic
          values = torch.FloatTensor(values)
          Qvals = torch.FloatTensor(Qvals)
          log_probs = torch.stack(log_probs)
          
          advantage = Qvals - values
          actor_loss = (-log_probs * advantage).mean()
          critic_loss = 0.5 * advantage.pow(2).mean()
          ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

          ac_optimizer.zero_grad()
          ac_loss.backward()
          ac_optimizer.step()
      
      model_path_save = os.path.join(script_path, 'ac_cb_{}_{}_{}.pt'.format(inter,date_save,0))
      save_model(actor_critic,model_path=model_path_save,Actor="Actor")

    model_path_save = os.path.join(script_path, 'ac_ended_{}_{}.pt'.format(date_save,0))
    save_model(actor_critic,model_path=model_path_save,Actor="Actor")
    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()
    return(actor_critic)


def get_model(model_path=''):
    '''
    Load `model` from disk. Location is specified in `model_path`. 
    '''
    model_class, model_state_dict = torch.load(model_path)
    #ActorCritic(2000, env.action_space.n,hidden_size).to(device)
    model = ActorCritic(2000, 5,1096).to(device)
    model.load_state_dict(model_state_dict)
    return model


def mainTest(agent):
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
              value, policy_dist = agent.forward(state)
              action = torch.argmax(policy_dist, dim=1)
              #print(action)
              #print(value,policy_dist) 
              next_state, reward, done, info = env.step(int(action))
              full_state = {
                  'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                  'done': done, 'info': info
              }
              #agent.update(**full_state)
              state = next_state
              episode_rewards += reward
              if done:
                  break
          if t > 0:
            rewards.append(episode_rewards)
      avg_rewards = sum(rewards)/len(rewards)
      print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
      return avg_rewards

  def timed_test(task,agent):
      start_time = time.time()
      rewards = []
      for tc in task['testcases']:
          
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
  timed_test(task,agent)

if __name__ == "__main__":
    env = construct_task2_env(4,4)
    
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    parser.add_argument('--test', dest='test', action='store_true', help='train the agent')
    parser.add_argument('--pretrain', dest='pretrain', action='store_true', help='train the agent')
    parser.add_argument('--path', dest='path', help='train the agent')
    parser.add_argument('--savepath', dest='savepath', help='train the agent')
    args = parser.parse_args()


    if args.train:
        model =  a2c(env) 
    elif args.pretrain:
        full_path = args.path
        save_path = args.savepath
        model_p = get_model(model_path=full_path)
        #model_p = train(ConvDQN, env,name_save=save_path,pretrain=True,model_pretrain=model_p)
        #save_model(model_p)
        model = model_p
    else:
        FAST_DOWNWARD_PATH = "/fast_downward/"
        #model_full_path = os.path.join(script_path, 'model_{}.pt'.format("19 05 42"))
        full_path = args.path
        print(full_path)
        model = get_model(model_path=full_path)
        #model_full_path_after = os.path.join(script_path, 'model_{}_{}.pt'.format("19 05 42",2))
        #model_post_train = train(model, env,name_save=model_full_path_after)
        #model = model_post_train
    mainTest(model)





