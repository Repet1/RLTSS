import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import make_interp_spline

# 超参数设置
EPISODES = 3000
STATE_SIZE = 3000
ACTION_SIZE = 100
GAMMA = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 48
MEMORY_SIZE = 2000
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
THRESHOLD = 40  # 奖励的阈值

episode_rewards = []
episode_losses = []


class PowerTraceEnv:
    def __init__(self, trace):
        self.trace = trace
        self.length = len(trace)
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.reset()
        self.previous_segments = []  # 存储已获取的迹线段

    def reset(self):
        self.current_pos = 0
        self.previous_segments = []  # 每次重置时清空
        return self.get_state()

    def step(self, action):
        distance = (action + 1) * 100
        next_pos = min(self.current_pos + distance, self.length - self.state_size)
        print(self.current_pos, ":", next_pos)
        current_segment = self.trace[self.current_pos: next_pos]
        current_segment = self.interpolate_segment(current_segment)

        # 计算奖励
        reward = self.calculate_reward(current_segment)

        done = next_pos >= self.length - self.state_size
        self.current_pos = next_pos
        return self.get_state(), reward, done

    def interpolate_segment(self, segment):
        """ 使用二阶 B 样条插值来统一段的长度 """
        x = np.arange(len(segment))
        spline = make_interp_spline(x, segment, k=2)  # 二阶 B 样条
        x_new = np.linspace(0, len(segment) - 1, self.state_size)
        return spline(x_new)

    def calculate_reward(self, current_segment):
        if not self.previous_segments:
            self.previous_segments.append(current_segment)
            return 0.0  # 首次没有奖励

        distances = []
        for prev_segment in self.previous_segments:
            prev_segment = self.interpolate_segment(prev_segment)
            distance = np.linalg.norm(current_segment - prev_segment)
            distances.append(distance)

        average_distance = np.mean(distances)

        # 打印调试信息
        print(f"Average Distance: {average_distance}")

        # 调整奖励逻辑
        if average_distance < THRESHOLD:
            reward = 1.0 - (average_distance / THRESHOLD)  # 根据距离调整奖励
        else:
            reward = -1.0  # 距离较大，给予惩罚

        self.previous_segments.append(current_segment)  # 更新上一个段
        return reward

    def get_state(self):
        local_state = self.trace[self.current_pos: self.current_pos + self.state_size]
        global_mean = np.mean(self.trace)
        global_std = np.std(self.trace)
        return np.append(local_state, [global_mean, global_std])


def build_model(state_size, action_size):
    model = tf.keras.Sequential()
    model.add(layers.Dense(80, input_dim=state_size + 2, activation='relu'))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.model = build_model(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def generate_synthetic_trace():
    trace = []
    with open('D:\\AliDownload\\MLP(2)\\BMLP\\权重和输入\\output.csv', 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        first_row = next(reader, None)
        if first_row is not None:
            trace = [float(value) for value in first_row]
    return np.array(trace[1251:len(trace)])


if __name__ == "__main__":
    trace = generate_synthetic_trace()
    env = PowerTraceEnv(trace)
    agent = DQNAgent(env.state_size, env.action_size)

    split_points = []

    for episode in range(EPISODES):
        state = env.reset().reshape(1, -1)
        total_reward = 0
        episode_splits = []
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = next_state.reshape(1, -1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            episode_splits.append(env.current_pos)
            if done:
                split_points.append(episode_splits)
                print(f"Episode: {episode + 1}/{EPISODES}, Reward: {total_reward}")
                break
        agent.replay()

    # 绘制训练曲线
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards Over Time")
    plt.show()

    plt.plot(episode_losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Episode Losses Over Time")
    plt.show()

    print(f"分割点信息：{split_points}")

    # 可视化分割点
    plt.plot(trace, label="Power Trace")
    for splits in split_points[-1]:  # 只绘制最后一个episode的分割点
        plt.axvline(x=splits, color='r', linestyle='--', label="Split Points" if splits == split_points[-1][0] else "")

    plt.title("Power Trace with Detected Split Points")
    plt.legend()
    plt.show()
