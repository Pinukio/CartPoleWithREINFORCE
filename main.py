# 바닥부터 배우는 강화 학습 P.228 REINFORCE 구현

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
    
    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            # Return
            R = r + gamma * R
            # prob: Policy에 의한 Action 선택 확률
            loss = -R * torch.log(prob)
            # Gradient 계산
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()[0]
        done = False

        while not done:
            prob = pi.forward(torch.from_numpy(s).float())
            # Categorical.sample()은 확률 분포에 의한 Sampling을 진행해 준다.
            # Episode 진행 과정에서 Policy에 의한 Action 선택임
            m = Categorical(prob)
            action = m.sample()
            s_prime, reward, done, trun, _ = env.step(action.item())
            # Episode가 진행되는 동안 Data를 수집함
            pi.put_data((reward, prob[action]))
            s = s_prime
            score += reward

        pi.train_net()
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode: {}, avg score: {}".format(n_epi, score/print_interval))
            score = 0.0
        env.close()