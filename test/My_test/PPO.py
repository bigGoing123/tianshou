import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
import warnings

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make('CartPole-v1',render_mode="human")
train_envs = DummyVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(20)])
test_envs = DummyVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(10)])
# net is the shared head of the actor and the critic
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128], device=device)
actor = Actor(net, action_shape=action_shape, device=device).to(device)
critic = Critic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
# optimizer of the actor and the critic
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)
dist = torch.distributions.Categorical
policy = PPOPolicy(actor=actor,
                   critic=critic,
                   optim=optim,
                   dist_fn=dist,
                   action_space=env.observation_space,
                   deterministic_eval=True)
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
test_collector = Collector(policy, test_envs)
result = OnpolicyTrainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=10,
    step_per_epoch=50000,
    repeat_per_collect=10,
    episode_per_test=10,
    batch_size=256,
    step_per_collect=2000,
    stop_fn=lambda mean_reward: mean_reward >= 195,
)