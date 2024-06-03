from armin_utils.RL.envs import game
from armin_utils.utils.tensor import tensor_to_numpy, to_tensor
from armin_utils.utils import clone_repo
import torch
import torch.nn as nn
from armin_utils.RL.utils import soft_update_params
import torch.optim as optim
from armin_utils.RL.reply_memory import ReplayMemory_Tuple
from collections import namedtuple
device='cpu'
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    
class DDPG:
    def __init__(self, action_dim, state_dim, gamma=0.99, tau=0.001):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim, 1).to(device).float()
        self.actor_target = Actor(state_dim, action_dim, 1).to(device).float()
        self.critic = Critic(state_dim, action_dim).to(device).float()
        self.critic_target = Critic(state_dim, action_dim).to(device).float()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)


    def calc_action(self, state, action_noise=None):
        if len(state.size()) == 1:
            state = torch.tensor(state).unsqueeze(0)
        x = state.to(device).float()
        self.actor.eval()
        mu = self.actor(x)
        self.actor.train()
        mu = mu.data
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            mu += noise
        mu = mu.clamp(-1, 1)
        return mu

    def update_params(self, batch):
        
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())
        
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values
        
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        soft_update_params(self.actor_target, self.actor, self.tau)
        soft_update_params(self.critic_target, self.critic, self.tau)
        
        return value_loss.item(), policy_loss.item()


    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()


env = game('C:/Users/Armin/Desktop/samples/')
agent = DDPG(state_dim=17, action_dim=6)
memory = ReplayMemory_Tuple(10000, saving_tensor=True)
batch_size = 128
Transition = namedtuple('Transition',('state', 'action', 'done', 'next_state', 'reward'))

for episode in range(500):
    epoch_return = 0
    timestep = 0
    state = env.reset()['observation']
    state = to_tensor(state)

    while timestep <= 300:
        action = agent.calc_action(state, None)
        action = tensor_to_numpy(action)[0]
        res = env.step(action)
        state = to_tensor(state)
        action = to_tensor(action)
        next_state = to_tensor(res['observation'])
        reward = to_tensor(res['reward'])
        if res['terminated'] == True:
            done = 1
        else:
            done = 0
        done = to_tensor(done)
        timestep += 1
        epoch_return += reward
        memory.push(state, action, done, next_state, reward)
        state = next_state
        if len(memory) > batch_size:
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))
            value_loss, policy_loss = agent.update_params(batch)
    print('Epoch_Return: {}'.format(str(epoch_return.item())))
    env.shots_to_video()
            
#%%
epoch_return

