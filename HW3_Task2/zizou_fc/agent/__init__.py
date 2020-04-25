try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import random

from . import dqn as dqn_model


class DQN(Agent):
    def __init__(self, *args, **kwargs):
    	self.model = dqn_model.get_model()
    
    def step(self, state, *args, **kwargs):
        return self.model.act(state)

def create_agent(test_case_id, *args, **kwargs):
    return DQN()


if __name__ == '__main__':
    import sys
    import time
    from env import construct_task2_env
    
    # import argparse
    # parser = argparse.ArgumentParser(description='Test DQN agent.')
    # parser.add_argument('--save', help='saved model path')
    # args = parser.parse_args()

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        for run in range(runs):
            state = env.reset()
            agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
            # agent.initialize(**agent_init)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)   
                next_state, reward, done, info = env.step(action)
                # full_state = {
                #     'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                #     'done': done, 'info': info
                # }
                # agent.update(**full_state)
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
            'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': 1000, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task()
    timed_test(task)