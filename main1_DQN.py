import gym
from main1_Agent import Agent_DQN

env = gym.make('CartPole-v0')
env = env.unwrapped

print("action_size:", env.action_space.n)
print("state_size:", env.observation_space.shape[0])

agent = Agent_DQN(action_size=env.action_space.n,
                  state_size=env.observation_space.shape[0])

total_steps = 0
total_episode = 190
reward_history = []
render = False

for e in range(total_episode):
    s = env.reset()
    episode_reward_sum = 0
    while True:
        if render:
            env.render()
        action = agent.get_action(s)
        s_, _ , done, info = env.step(action)

        x, _, theta, _ = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians
        reward = r1 + r2
        agent.store_transition(s, action, reward, s_)
        episode_reward_sum += reward
        agent.learn()

        if done:
            print(e,"-th episode, reward sum: ", episode_reward_sum,
                    'learning rate:', agent.learning_rate_history[len(agent.learning_rate_history)-1],
                    ' epsilon: ', agent.epsilon)
            reward_history.append(episode_reward_sum)
            if episode_reward_sum > 3000:
                render = True
            break

        s = s_
        total_steps += 1

agent.plot_loss()
agent.plot_reward(reward_history)
