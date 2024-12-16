import numpy as np  # 导入NumPy库，用于数值计算
import random  # 导入随机模块，用于随机选择动作
from collections import deque  # 导入双向队列，用于经验回放
import tensorflow as tf  # 导入TensorFlow库，用于构建深度学习模型
from tensorflow.keras import layers  # 导入Keras中的layers模块，用于构建神经网络层
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘制图像
import csv  # 导入csv库，用于读取CSV文件

# 超参数设置
EPISODES = 10000  # 总共训练的次数
STATE_SIZE = 2000  # 每次输入到DQN的状态长度（局部窗口大小）
ACTION_SIZE = 100  # 动作空间大小，表示选择不同的分割距离
GAMMA = 0.95  # 折扣率，控制未来奖励的影响程度
LEARNING_RATE = 0.001  # 学习率，控制模型更新的步幅
BATCH_SIZE = 48  # 每次训练时从经验池中抽取的样本数
MEMORY_SIZE = 2000  # 经验池的最大容量
EPSILON_DECAY = 0.995  # 探索率的衰减速率，逐渐减少探索行为
EPSILON_MIN = 0.01  # 探索率的最小值

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
        self.reset()  # 初始化环境

    def reset(self):
        self.current_pos = 0  # 每次重置时，当前分割位置重置为0
        return self.get_state()  # 返回初始状态

    def step(self, action):
        # 根据动作计算新的分割位置，每次移动30个点，避免频繁分割
        distance = (action + 1) * 100
        next_pos = min(self.current_pos + distance, self.length - self.state_size)

        # 计算当前段的均值和标准差作为奖励的一部分
        current_segment = self.trace[self.current_pos: next_pos]
        segment_mean = np.mean(current_segment)
        segment_std = np.std(current_segment)

        # 奖励由两部分组成：与全局均值的相似性和分割的惩罚
        global_mean = np.mean(self.trace)
        similarity_reward = -abs(segment_mean - global_mean)  # 均值差异越小，奖励越高
        over_split_penalty = 0.5 * (distance / self.length)  # 过多分割的惩罚

        reward = similarity_reward - over_split_penalty  # 总体奖励

        done = next_pos >= self.length - self.state_size  # 检查是否到达结束位置
        self.current_pos = next_pos  # 更新当前分割位置
        return self.get_state(), reward, done  # 返回新状态、奖励和是否结束

    def get_state(self):
        # 返回当前窗口的局部状态，并附加全局均值和标准差
        local_state = self.trace[self.current_pos: self.current_pos + self.state_size]
        global_mean = np.mean(self.trace)  # 全局均值
        global_std = np.std(self.trace)  # 全局标准差
        return np.append(local_state, [global_mean, global_std])  # 将全局特征添加到局部状态中


# 构建DQN的神经网络模型
def build_model(state_size, action_size):
    model = tf.keras.Sequential()  # 初始化顺序模型
    model.add(layers.Dense(128, input_dim=state_size + 2, activation='relu'))  # 输入层，带ReLU激活函数
    model.add(layers.Dense(128, activation='relu'))  # 隐藏层，带ReLU激活函数
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))  # 输出层，输出动作的值
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))  # 使用均方误差损失和Adam优化器
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
            return random.randrange(self.action_size)  # 随机选择动作
        act_values = self.model.predict(state)  # 根据当前状态预测动作值
        return np.argmax(act_values[0])  # 返回具有最高值的动作

    def replay(self):
        # 从经验池中随机采样进行训练
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward  # 默认目标值为奖励
            if not done:#如果done不是true代表不是中止状态
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])  # 计算目标Q值=即时奖励加上未来最优动作的折扣奖励
            target_f = self.model.predict(state)  # 获取当前状态 state 下所有动作的 Q 值。
            target_f[0][action] = target  # 更新选择的动作的Q值
            self.model.fit(state, target_f, epochs=1, verbose=0)  # 使用目标值训练模型
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # 每次训练后减少epsilon，减少随机探索


# 读取功耗曲线数据
def generate_synthetic_trace():
    trace = []
    start_index = 0
    end_index = 1000  # 这个变量在这个函数中没有被使用
    with open('D:\\AliDownload\\MLP(2)\\BMLP\\权重和输入\\output.csv', 'r', encoding='utf-8-sig') as file:
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

    for episode in range(EPISODES):
        state = env.reset().reshape(1, -1)  # 重置环境并获取初始状态
        total_reward = 0  # 记录该episode的总奖励
        episode_splits = []  # 记录每个episode的分割点
        for time in range(500):
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
        loss = agent.replay()  # 训练智能体
        if loss is not None:
            episode_losses.append(loss)
        episode_rewards.append(total_reward)  # 保存每个episode的奖励

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

