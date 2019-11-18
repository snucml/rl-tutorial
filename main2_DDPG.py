import gym
from main2_Agent import Agent_DDPG

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

total_episodes = 100
total_episode_steps = 200

agent = Agent_DDPG(a_dim, s_dim, a_bound)
render = False
episode_reward_history = []

for i in range(total_episodes):
    s = env.reset()
    episode_reward = 0
    for j in range(total_episode_steps):
        if render:
            env.render()
        a = agent.choose_action(s)
        s_, r, done, info = env.step(a)
        agent.store_transition(s, a, r / 10, s_)
        agent.learn()
        s = s_
        episode_reward += r
        if j == total_episode_steps-1:
            print('Episode:', i, ' Reward: %i' % int(episode_reward))
            episode_reward_history.append(int(episode_reward))
            if episode_reward > -100:
                render = True
            break

agent.plot_loss()
agent.plot_reward(episode_reward_history)