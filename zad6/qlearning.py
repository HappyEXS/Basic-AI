import gym
import random
import numpy as np
from time import sleep
from IPython.display import clear_output


def qlearning(env, episodes, max_steps, exploration, learning_rate, discount):
    '''       env - learning environment
    episodes      - episodes performed in learning process
    max_steps     - max number of steps in each episode
    exploration   - parameter of exploration (more means more exploration)
    learning_rate - level of learning in each step
    discount      - importance of reward in learning process
    '''
    qtable = np.zeros((500, 6), dtype=np.double)    # Table hard coded for Taxi-v3

    all_epochs, all_rewards = [], []

    for episode in range(episodes):
        env.reset()

        epochs, total_reward = 0, 0

        for step in range(max_steps):
            state = env.s

            # Choose action in exploration or exploitation mode
            if random.random() > exploration:
                action = random.randint(0, 5)
            else:
                action = np.argmax(qtable[state])

            # Perform chosen action
            next_state, reward, done, info = env.step(action)

            # Update qtable - greedy strategy
            qtable[state][action] = ((1-learning_rate) * qtable[state][action]) \
                + (learning_rate * (reward + discount*max(qtable[next_state])))

            epochs += 1
            total_reward += reward
            if done:
                break

        all_epochs.append(epochs)
        all_rewards.append(total_reward)

    return qtable, np.average(all_epochs), np.average(all_rewards)


def test_qtable(env, qtable, episodes, max_steps):
    '''Tests qtable function for number of episodes with max_step'''
    all_epochs, all_rewards = [], []
    penalties, completed = 0, 0

    for episode in range(episodes):
        env.reset()
        epochs, total_reward = 0, 0

        for step in range(max_steps):
            state = env.s
            action = np.argmax(qtable[state])
            state, reward, done, info = env.step(action)

            total_reward += reward
            epochs += 1
            if reward == -10:
                penalties += 1
            if done:
                completed +=1
                break

        all_epochs.append(epochs)
        all_rewards.append(total_reward)

    return np.average(all_epochs), np.average(all_rewards), completed*100/episodes, penalties

def animation(env, qtable):
    '''Performs visualization of algorithm'''
    env.reset()
    epochs, penalties, reward = 0, 0, 0
    frames = []
    done = False
    while not done:
        state = env.s
        action = np.argmax(qtable[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1
    return frames

def print_frames(frames):
    '''Prints frames from animation()'''
    i = 0
    for frame in frames:
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Iterations: {i}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        i += 1

def test_parameters(env):
    '''Test episodes'''
    print("Number of episodes; avg_steps; avg_reward; percentage of success; penalties")
    for episode in [100, 250, 500, 1000, 5000, 10000, 50000, 100000]:
        table, ep, rew = qlearning(env, episodes=episode, max_steps=500, exploration=0.8, learning_rate=1, discount=0.8)
        ep, rew, com, pen =test_qtable(env, table, 1000, 200)
        print(f"{episode}\t\t{ep}\t\t{rew}\t\t{com}%\t\t{pen}")

    '''Test exploration'''
    print("exploration; avg_steps; avg_reward; percentage of success; penalties")
    for exploration in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1]:
        table, ep, rew = qlearning(env, episodes=5000, max_steps=500, exploration=exploration, learning_rate=1, discount=0.8)
        ep, rew, com, pen = test_qtable(env, table, 1000, 200)
        print(f"{exploration}\t\t{ep}\t\t{rew}\t\t{com}%\t\t{pen}")

    '''Test learning_rate'''
    print("learning_rate; avg_steps; avg_reward; percentage of success; penalties")
    for lr in [0.1, 0.5, 0.7, 1, 1.2, 1.6, 2]:
        table, ep, rew = qlearning(env, episodes=5000, max_steps=500, exploration=0.8, learning_rate=lr, discount=0.8)
        ep, rew, com, pen = test_qtable(env, table, 1000, 200)
        print(f"{lr}\t\t{ep}\t\t{rew}\t\t{com}%\t\t{pen}")

    '''Test discount'''
    print("discount; avg_steps; avg_reward; percentage of success; penalties")
    for discount in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1]:
        table, ep, rew = qlearning(env, episodes=5000, max_steps=500, exploration=0.8, learning_rate=1, discount=discount)
        ep, rew, com, pen = test_qtable(env, table, 1000, 200)
        print(f"{discount}\t\t{ep}\t\t{rew}\t\t{com}%\t\t{pen}")

def main():
    env = gym.make("Taxi-v3").env
    # test_parameters(env)

    table, learn_ep, learn_rew = qlearning(env, episodes=5000, max_steps=500, exploration=0.8, learning_rate=1, discount=0.8)
    print_frames(animation(env, table))
    ep, rew, com, pen = test_qtable(env, table, 1000, 200)
    print("\nEpisodes; avg_steps; avg_reward; percentage of success; penalties")
    print(f"{5000}\t\t{ep}\t\t{rew}\t\t{com}%\t\t{pen}")

if __name__ == "__main__":
    main()
