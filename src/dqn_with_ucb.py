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
from environment_stage_1 import Env
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
import math
from collections import deque

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


#---Define parameters of the network---#
action_dim = 5
state_dim = 26
hidden_dim = 300
action_w_max = 2. #-rad/s-#

def choose_w_action_velocity(action, w_max=action_w_max, a_dim=action_dim):
    '''
        Function that chooses the angular(w) to the turtlebot3 given the action dimension

        param:
            action: action choosed by the network
            w_max: maximum angular velocity of the robot
            a_dim: the number of discritized actions of the agent

        return: velocity of the robot in rad/s 
    '''
    return w_max*(action/((a_dim-1.)/2.) - 1.)

