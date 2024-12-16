import numpy as np  # 导入NumPy库，用于数值计算
import random  # 导入随机模块，用于随机选择动作
from collections import deque  # 导入双向队列，用于经验回放
import tensorflow as tf  # 导入TensorFlow库，用于构建深度学习模型
from tensorflow.keras import layers  # 导入Keras中的layers模块，用于构建神经网络层
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘制图像
import csv  # 导入csv库，用于读取CSV文件
from scipy.interpolate import make_interp_spline  # 导入样条插值
from tqdm import tqdm

# 超参数设置
EPISODES = 3000  # 总共训练的次数
STATE_SIZE = 55  # 每次输入到DQN的状态长度（局部窗口大小）
ACTION_SIZE = 65  # 动作空间大小，表示选择不同的分割距离
GAMMA = 0.95  # 折扣率，控制未来奖励的影响程度
LEARNING_RATE = 0.05  # 学习率，控制模型更新的步幅
BATCH_SIZE = 100  # 每次训练时从经验池中抽取的样本数
MEMORY_SIZE = 2000  # 经验池的最大容量
EPSILON_DECAY = 0.9999  # 探索率的衰减速率，逐渐减少探索行为
EPSILON_MIN = 0.01  # 探索率的最小值
THRESHOLD = 0.4 # 奖励的阈值

# 用于存储每个episode的奖励和损失
episode_rewards = []  # 每个episode的奖励
episode_losses = []  # 每个episode的损失


# 环境类，模拟处理功耗曲线的分割问题
class PowerTraceEnv:
    def __init__(self, trace):
        self.trace = trace  # 功耗曲线
        self.length = len(trace)  # 曲线的长度

        self.state_size = STATE_SIZE  # 状态的大小（窗口大小）
        self.action_size = ACTION_SIZE  # 动作的大小
        self.previous_segments = []  # 存储已获取的迹线段
        self.reset()  # 初始化环境

    def reset(self):
        self.current_pos = 0  # 每次重置时，当前分割位置重置为0
        self.previous_segments = []  # 清空之前的段
        return self.get_state()  # 返回初始状态

    def step(self, action):
        # 根据动作计算新的分割位置

        next_pos = min(self.current_pos + action,  self.length - self.state_size)
        if  self.length - self.state_size-next_pos<3:
          #  next_pos=self.current_pos + action
            self.current_pos=self.length - self.state_size-action

        # 获取当前段

        current_segment = self.trace[self.current_pos: next_pos]
        current_segment = self.interpolate_segment(current_segment)

        # 计算奖励
        reward = self.calculate_reward(current_segment)

        done = next_pos >= self.length - self.state_size  # 检查是否到达结束位置
        self.current_pos = next_pos  # 更新当前分割位置
        print(self.get_state())

        return self.get_state(), reward, done  # 返回新状态、奖励和是否结束

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
        std_devs = []

        for prev_segment in self.previous_segments:
            prev_segment = self.interpolate_segment(prev_segment)
            distance = np.linalg.norm(current_segment - prev_segment)  # 欧几里得距离
            distances.append(distance)

            # 计算标准差
            std_dev_diff = np.abs(np.std(current_segment) - np.std(prev_segment))
            std_devs.append(std_dev_diff)

        average_distance = np.mean(distances)*0.1  # 平均距离
        average_std_dev_diff = np.mean(std_devs)  # 平均标准差差异

        # 设定奖励：结合均值差异和标准差
        similarity_reward = -np.log(average_distance + average_std_dev_diff)  # 距离和标准差越小，奖励越高

        if similarity_reward > THRESHOLD:
            return similarity_reward  # 以阈值为基础的奖励
        else:
            return -1.0  # 距离较大，给予惩罚

    def get_state(self):
        # 返回当前窗口的局部状态，并附加全局均值和标准差
        local_state = self.trace[self.current_pos: self.current_pos + self.state_size]
        global_mean = np.mean(self.trace)  # 全局均值
        global_std = np.std(self.trace)  # 全局标准差
        return np.append(local_state, [global_mean, global_std])  # 将全局特征添加到局部状态中
    def get_intistate(self):
        # 返回当前窗口的局部状态，并附加全局均值和标准差
        local_state = self.trace[0: 41]
        global_mean = np.mean(local_state)  # 全局均值
        global_std = np.std(local_state)  # 全局标准差
        return np.append(local_state, [global_mean, global_std])  # 将全局特征添加到局部状态中

# 构建DQN的神经网络模型

def build_model(state_size, action_size):
    inputs = tf.keras.Input(shape=(state_size + 2,))
    x = tf.keras.layers.Dense(40, activation='relu')(inputs)
    x = tf.keras.layers.Dense(40, activation='relu')(x)
    outputs = tf.keras.layers.Dense(action_size, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model

# DQN智能体类
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 状态大小
        self.action_size = action_size  # 动作空间大小
        self.memory = deque(maxlen=MEMORY_SIZE)  # 使用双向队列存储经验池
        self.gamma = GAMMA  # 折扣率
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = EPSILON_DECAY  # 探索率衰减
        self.epsilon_min = EPSILON_MIN  # 最小探索率
        self.model = build_model(state_size, action_size)  # 构建DQN模型

    def remember(self, state, action, reward, next_state, done):
        # 将经验存储到经验池中
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # epsilon-greedy 策略：以epsilon的概率选择随机动作，以(1-epsilon)的概率选择最优动作
        if np.random.rand() <= self.epsilon:
            # 随机选择动作，从1开始到self.action_size-1结束

            return random.randrange(5, self.action_size)  # 随机选择动作
        act_values = self.model.predict(state)  # 根据当前状态预测动作值

        return np.argmax(act_values[0])  # 返回具有最高值的动作

    def replay(self):
        # 从经验池中随机采样进行训练
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward  # 默认目标值为奖励
            if not done:  # 如果done不是true代表不是中止状态
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])  # 计算目标Q值
            print("replay:",state)
            target_f = self.model.predict(state)  # 获取当前状态下所有动作的Q值。
            target_f[0][action] = target  # 更新选择的动作的Q值
            self.model.fit(state, target_f, epochs=1, verbose=0)  # 使用目标值训练模型
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 每次训练后减少epsilon，减少随机探索


def generate_synthetic_trace():
    trace = []
    with open('D:\\项目\\MLP\\权重和输入\\qhdata.csv', 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        first_row = next(reader, None)  # 获取第一行
        if first_row is not None:
            # 假设CSV文件的第一行有多个以逗号分隔的数值
            trace = [float(value) for value in first_row]  # 将第一行的每个值转换为浮点数
    return np.array(trace)  # 返回读取的数据


# 主函数

if __name__ == "__main__":
    trace = generate_synthetic_trace()  # 生成功耗曲线
    env = PowerTraceEnv(trace)  # 创建环境
    agent = DQNAgent(env.state_size, env.action_size)  # 创建DQN智能体

    split_points = []  # 存储每个episode的分割点

    # 使用tqdm创建进度条
    for episode in range(EPISODES):
        state = env.reset().reshape(1, -1)  # 重置环境并获取初始状态
        total_reward = 0  # 记录该episode的总奖励
        episode_splits = []  # 记录每个episode的分割点
        for time in range(10000):
            action = agent.act(state)  # 智能体在当前状态下选择动作
            next_state, reward, done = env.step(action)  # 执行动作并获取新的状态和奖励

            next_state = next_state.reshape(1, -1)
            agent.remember(state, action, reward, next_state, done)  # 记录经验

            state = next_state  # 更新当前状态
            total_reward += reward  # 累加奖励

            episode_splits.append(env.current_pos)  # 记录当前的分割点
            if done:  # 如果到达结束状态

                split_points.append(episode_splits)  # 保存分割点
                print(f"Episode: {episode + 1}/{EPISODES}, Reward: {total_reward}")
                break
        agent.replay()  # 训练智能体

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

    print(f"分割点信息：{split_points}")  # 打印每个episode的分割点

    # 可视化分割点
    plt.plot(trace, label="Power Trace")
    print(split_points)
    for splits in split_points[-1]:  # 只绘制最后一个episode的分割点

        plt.axvline(x=splits, color='r', linestyle='--', label="Split Points" if splits == split_points[-1][0] else "")

    plt.title("Power Trace with Detected Split Points")
    plt.legend()
    plt.show()
