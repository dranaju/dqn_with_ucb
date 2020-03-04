#!/usr/bin/env python

"""
   Author: Dranaju (Junior Costa de Jesus)

   Code of DQN with the UCB exploration system
   for navigation of mobile robots (Turtlebot3)
"""

#---Libraries implemented---#
import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
from environment_stage import Env
import torch
import torch.nn.functional as F
import torch.optim as optim
import gc
import torch.nn as nn
import math
from collections import deque
import copy

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

#---Replay Memory for the DQN---#
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push_to_memory(self, state, action, reward, state_plus_1):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = state, action, reward, state_plus_1
        self.position = (self.position + 1) % self.capacity
        
    def pull_from_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, state_plus_1= map(np.stack, zip(*batch))
        return state, action, reward, state_plus_1
    
    def __len__(self):
        return len(self.memory)

#---DQN implemented with mish activation function---#
class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, init_w=3e-3):
        super(DQN, self).__init__()
        
        self.linear1 = nn.Linear(in_features=state_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        
        #self.linear1.weight.data.uniform_(-init_w, init_w)
        #self.linear1.bias.data.uniform_(-init_w, init_w)
        #self.linear2.weight.data.uniform_(-init_w, init_w)
        #self.linear2.bias.data.uniform_(-init_w, init_w)
        #self.linear3.weight.data.uniform_(-init_w, init_w)
        #self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def mish(self, x):
        '''
            Mish: A Self Regularized Non-Monotonic Neural Activation Function
            https://arxiv.org/abs/1908.08681v1
            implemented for PyTorch / FastAI by lessw2020
            https://github.com/lessw2020/mish

            param:
                x: output of a layer of a neural network

            return: mish activation function
        '''
        return x*(torch.tanh(F.softplus(x)))
        
    def forward(self, state):
        state = torch.FloatTensor(state)
        x = self.mish(self.linear1(state))
        x = self.mish(self.linear2(x))
        x = self.linear3(x)
        return x

#---Choose device for torch---#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\n---\nPytorch is using the {} as device \n---\n'.format(device))

#---Function that makes the update of the DQN---#
#-Loss function used in the network-#
loss_function = nn.MSELoss()
def dqn_update(batch_size,
                 gamma=0.99,
                 tau=0.001):
    '''
        Function that updates the DQN

        param:
            batch_size: size of the buffer of memory
            gamma: discont value of gamma
            tau: value to make the soft-update of the DQN target
    '''
    state, action, reward, state_plus_1 = replay_buffer.pull_from_memory(batch_size)
    
    state      = torch.FloatTensor(state)
    state_plus_1 = torch.FloatTensor(state_plus_1)
    action     = torch.LongTensor(np.reshape(action, (batch_size, 1)))
    reward     = torch.FloatTensor(reward).unsqueeze(1)
    
    predicted_q_value = dqn_net.forward(state)
    predicted_q_value = predicted_q_value.gather(1,action)
    q_value_plus_1_target = dqn_target_net.forward(state_plus_1).detach()
    max_q_value_plus_1_target = q_value_plus_1_target.max(1)[0].unsqueeze(1)
    expected_q_value = reward + gamma*max_q_value_plus_1_target
    
    loss = loss_function(predicted_q_value, expected_q_value)
    
    dqn_optimizer.zero_grad()
    loss.backward()
    dqn_optimizer.step()
    
    for target_param, param in zip(dqn_target_net.parameters(), dqn_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )   

#---Define parameters of the network---#
action_dimension = 5
state_dimension = 26
hidden_dimension = 300
action_w_max = 2. #-rad/s-#
action_v_constant = 1.5 #-m/s-#
learning_rate = 0.001
replay_memory_size = 20000
world = 'stage_1'
batch_size  = 128

#---Implementation of the UCB---#
#-Constant of UCB exploration system-#
c_constant = 2
#-Array of the the dimension of actions to represent how many time they were chosed-#
number_times_action_selected = np.zeros(action_dimension)
def ucb_exploration(action, episode):
    '''
        Function of the UCB exploration system

        param:
            action: q-value of the actions
            episode: episode of training for the function

        return: q-value of the action with UCB exploration system
    '''
    print('ucb', c_constant*np.sqrt(np.log(episode + 0.1)/(number_times_action_selected + 0.1)))
    return np.argmax(action + c_constant*np.sqrt(np.log(episode + 0.1)/(number_times_action_selected + 0.1)))

#---Objects of the DQN and target_network---#
dqn_net = DQN(state_dim=state_dimension, hidden_dim=hidden_dimension, action_dim=action_dimension)
dqn_target_net = DQN(state_dim=state_dimension, hidden_dim=hidden_dimension, action_dim=action_dimension)

#---Copying the parameters of dqn and dqn_target_network---#
for target_param, param in zip(dqn_target_net.parameters(), dqn_net.parameters()):
    target_param.data.copy_(param.data)

#---defining the optimizer of the network---#
dqn_optimizer  = optim.Adam(dqn_net.parameters(), lr=learning_rate)

#---inicializing the replay memory---#
replay_buffer = ReplayMemory(replay_memory_size)

#---function that chooses the angular velicity of the turtlebot---#
def choose_w_action_velocity(action, w_max=action_w_max, a_dim=action_dimension):
    '''
        Function that chooses the angular(w) to the turtlebot3 given the action dimension

        param:
            action: action choosed by the network
            w_max: maximum angular velocity of the robot
            a_dim: the number of discritized actions of the agent

        return: velocity of the robot in rad/s 
    '''
    return w_max*(action/((a_dim-1.)/2.) - 1.)

#---Save function of the network---#
def save_models(episode_count):
    '''
        Saves the models the DQN

        param:
            episode_count: episode taht will save
    '''
    torch.save(dqn_net.state_dict(), dirPath + '/models/' + world + '/'+str(episode_count)+ '_policy_net.pth')
    torch.save(dqn_target_net.state_dict(), dirPath + '/models/' + world + '/'+str(episode_count)+ 'value_net.pth')
    print("====================================")
    print("Model has been saved...")
    print("====================================")

#---Load function of the network---#
def load_models(episode):
    '''
        Loads the model of the DQN

        param:
            episode: loads the episode that were saved
    '''
    dqn_net.load_state_dict(torch.load(dirPath + '/models/' + world + '/'+str(episode)+ '_policy_net.pth'))
    dqn_target_net.load_state_dict(torch.load(dirPath + '/models/' + world + '/'+str(episode)+ 'value_net.pth'))
    print('***Models load***')

#---True if it is training and False if it is testing---#
is_training = True

ucb_exploration_use = False

#---loads episodes for the network parameters---#
'''
    (comment line if it is not using)
'''
# load_models(360)

#---parameters of the training---#
max_episodes  = 10001
max_steps   = 500
rewards     = []
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.1
exploration_decay_rate = 0.02

#---
print('State Dimensions: ' + str(state_dimension))
print('Action Dimensions: ' + str(action_dimension))
print('Action: constant linear velocity ' + str(1.5) + ' m/s and maximum angular velocity ' + str(action_w_max) + ' rad/s')

#---starts node and training---#
if __name__ == '__main__':
    rospy.init_node('dqn_node')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()
    how_memory_before_training = 2

    for ep in range(max_episodes):
        done = False
        state = env.reset()
        if is_training and ep%2 == 0 and len(replay_buffer) > how_memory_before_training*batch_size:
            print('Episode ' + str(ep) + ' training')
        else:
            print('Episode ' + str(ep) + ' evaluating')

        rewards_current_episode = 0.

        for step in range(max_steps):
            state = np.float32(state)
            # print('state', state[-2:])
            action = dqn_net.forward(state)

            if is_training and ep%2 == 0 and len(replay_buffer) > how_memory_before_training*batch_size:
                # print('exploring')
                if not ucb_exploration_use:
                    exploration_rate_threshold = random.uniform(0,1)
                    if exploration_rate_threshold > exploration_rate:
                        action = np.argmax(action.detach().numpy())
                    else:
                        action = random.randrange(action_dimension)
                else:
                    action = ucb_exploration(action.detach().numpy(), ep)
                    number_times_action_selected[action] += 1
            else:
                # print(('evaluating {}').format(len(replay_buffer)))
                action = np.argmax(action.detach().numpy())

            state_plus_1, reward, done = env.step(choose_w_action_velocity(action))
            # print('action', action,'r',reward)

            if ep%2 == 0 or not len(replay_buffer) > how_memory_before_training*batch_size:
                if reward == 100:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        replay_buffer.push_to_memory(state, action, reward, state_plus_1)
                else:
                    replay_buffer.push_to_memory(state, action, reward, state_plus_1)

            if len(replay_buffer) > how_memory_before_training*batch_size and is_training and ep%2 == 0:
                dqn_update(batch_size)

            rewards_current_episode += reward
            state_plus_1 = np.float32(state_plus_1)
            state = copy.deepcopy(state_plus_1)

            if done:
                break

        print('Reward per ep: ' + str(rewards_current_episode))
        
        rewards.append(rewards_current_episode)
        if len(replay_buffer) > how_memory_before_training*batch_size and not ep%2 == 0:
            result = rewards_current_episode
            pub_result.publish(result)
        if ep%2 == 0:
            exploration_rate = (min_exploration_rate +
                    (max_exploration_rate - min_exploration_rate)* np.exp(-exploration_decay_rate*ep))
        print('Exploration rate: ' + str(exploration_rate))
        if ep%20 == 0:
            save_models(ep)