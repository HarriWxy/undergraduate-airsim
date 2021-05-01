#!/usr/bin/env python
#============================ 导入所需的库 ===========================================
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout

import cv2
import sys
import random
import numpy as np
from collections import deque
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'
import datetime
import matplotlib.pyplot as plt

sys.path.append("game/")
import wrapped_flappy_bird as game

GAME = 'FlappyBird' # 游戏名称
ACTIONS = 2 # 2个动作数量
ACTIONS_NAME=['不动','起飞']  #动作名
GAMMA = 0.99 # 未来奖励的衰减
OBSERVE = 1000. # 训练前观察积累的轮数
EPSILON = 1
REPLAY_MEMORY = 3600 # 观测存储器D的容量
BATCH = 36 # 训练batch大小
TIMES = 500000

class MyNet(Model):
    def __init__(self):
        super().__init__()
        self.c1_1 = Conv2D(filters=32, kernel_size=(8, 8),strides=4, padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a1_1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.c2_1 = Conv2D(filters=64, kernel_size=(4, 4),strides=2, padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a2_1 = Activation('relu')  # 激活层
        self.c2_2 = Conv2D(filters=64, kernel_size=(3, 3),strides=1, padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        #self.b1 = BatchNormalization()  # BN层
        self.a2_2 = Activation('relu')  # 激活层
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
        self.f2 = Dense(ACTIONS, activation=None,
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))

    def call(self, x):
        #print(x.shape)
        x = self.c1_1(x)
        x = self.a1_1(x)
        x = self.d1(x)
        x = self.p1(x)

        x = self.c2_1(x)
        x = self.a2_1(x)
        x = self.c2_2(x)
        x = self.a2_2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        y = self.f2(x)
        return y

def trainNetwork(istrain):
#============================ 模型创建与加载 ===========================================

    # 模型创建
    net1 = MyNet()
    netstar = MyNet()
#============================ 配置模型 ===========================================
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-6, epsilon=1e-08)  #1e-6

    t = 0 #初始化TIMESTEP
    losses=[]
    scores=[]
    Loss=0
    Score=0

    # 加载保存的网络参数
    checkpoint_save_path = "./model/FlappyBird"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        net1.load_weights(checkpoint_save_path)
    else:
        print('-------------train new model-----------------')
    best_checkpoint_save_path = "./best/FlappyBird"
    if os.path.exists(best_checkpoint_save_path + '.index'):
        netstar.load_weights(best_checkpoint_save_path)
    rank_file_r = open("rank.txt","r")
    best = int(rank_file_r.readline())
    rank_file_r.close()
#============================ 加载(搜集)数据集 ===========================================

    # 打开游戏
    game_state = game.GameState()

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    D = deque()

    #初始化状态并且预处理图片，把连续的四帧图像作为一个输入（State）
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    s_t, r_0, terminal, _ = game_state.frame_step(do_nothing)
    #print(s_t.shape)

    # 开始训练
    while t < TIMES+1:
        # 根据输入的s_t,选择一个动作a_t
        epsilon = EPSILON - (EPSILON-0.1)*t/TIMES # 网络的过早介入会导致
        # 学习率
        learning_r=0.03-(0.03-0.00025)*t/TIMES
        optimizer=tf.keras.optimizers.RMSprop(learning_r,0.99,0.0,1e-7)
        
        a_t_to_game = np.zeros([ACTIONS])
        action_index = 0

        #贪婪策略，有episilon的几率随机选择动作去探索，否则选取Q值最大的动作
        if random.random() <= epsilon and istrain:
            print("----------Random Action----------")
            action_index = 0 if random.random() < 0.8 else 1
            a_t_to_game[action_index] = 1
        else:
            print("-----------net choice----------------")
            readout_t = net1(tf.expand_dims(tf.constant(s_t, dtype=tf.float32), 0))
            action_index = np.argmax(readout_t)
            print("-----------index----------------")
            print(action_index)
            a_t_to_game[action_index] = 1

        #执行这个动作并观察下一个状态以及reward
        s_t1, r_t, terminal, score = game_state.frame_step(a_t_to_game)
        print("============== score ====================")
        print(score)
        Score+=score

        
        #if score_one_round >= best:
        #    test = True

        if score > best:
            net1.save_weights(best_checkpoint_save_path)
            rank_file_w = open("rank.txt","w")
            rank_file_w.write("%d" % score)
            rank_file_w.close()
            print("********** best score updated!! *********")
            best = score
            netstar.set_weights(net1.get_weights())
        # if score >= best:
        #     f = open("scores.txt","a")
        #     f.write("========= %d ========== %d \n" % (t+old_time, score))
        #     f.close()

        # a_t = action_index

        s_t = tf.convert_to_tensor(s_t, dtype=tf.float32)
        action_index = tf.constant(action_index, dtype=tf.int32)
        r_t = tf.constant(r_t, dtype=tf.float32)
        s_t1 = tf.constant(s_t1, dtype=tf.float32)
        terminal = tf.constant(terminal, dtype=tf.float32)

        # 将观测值存入之前定义的观测存储器D中
        D.append((s_t, action_index, r_t, s_t1, terminal))
        #如果D满了就替换最早的观测
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 更新状态，不断迭代
        s_t = s_t1
        t += 1

#============================ 训练网络 ===========================================

        # 观测一定轮数后开始训练
        if (t > OBSERVE) and istrain:
            # 随机抽取minibatch个数据训练
            print("==================start train====================")
            
            minibatch = random.sample(D, BATCH)

            # 获得batch中的每一个变量
            b_s = []
            b_a = []
            b_r = []
            b_s_ = []
            b_done = []
            for d in minibatch:
                b_s.append(d[0])
                b_a.append(d[1])
                b_r.append(d[2])
                b_s_.append(d[3])
                b_done.append(d[4])

            b_s = tf.stack(b_s, axis=0)
            b_a = tf.expand_dims(b_a, axis=1) # 将一维的动作索引扩展成二维，每一个是独立的
            b_a = tf.stack(b_a, axis=0)
            b_r = tf.stack(b_r, axis=0)
            b_s_ = tf.stack(b_s_, axis=0)
            b_done = tf.stack(b_done, axis=0)

    

            # 训练
            with tf.GradientTape() as tape:
                q_output = net1(b_s)
                index = tf.expand_dims(tf.range(0, BATCH), 1) # index的二维向量
                # DDQN
                q_next = netstar(b_s_) # 每一行出一个最大值
                
                index_b_a = tf.concat((index, b_a), axis=1) # [batch_index,action_index]
                q = tf.gather_nd(q_output, index_b_a) # 当前状态下选择这个动作得到的Q值
                q_next = tf.gather_nd(q_next,index_b_a)

                q_truth = b_r + GAMMA * q_next* (tf.ones(BATCH) - b_done)
                
                loss = tf.losses.MSE(q_truth, q)
                Loss += loss
                print("loss = %f" % loss)
                gradients = tape.gradient(loss, net1.trainable_variables)
                optimizer.apply_gradients(zip(gradients, net1.trainable_variables))

            # 每1000轮保存一次网络参数
            if t % 1000 == 0:
                print("=================model save====================")
                net1.save_weights(checkpoint_save_path)
                netstar.set_weights(net1.get_weights())
                Score=Score/1000
                Loss=Loss/1000
                losses.append(Loss)
                scores.append(Score)
                Score=0
                Loss=0

        # 打印信息

        print("TIMESTEP", t, "|  ACTION", ACTIONS_NAME[action_index], "|  REWARD", r_t)#, \
        #    "|  Q_MAX %e \n" % np.max(readout_t))
        # write info to files
    
    plt.figure()
    plt.plot(losses)
    plt.xlabel("time")
    plt.ylabel('loss')
    plt.savefig('savefig'+datetime.datetime.now().strftime('%d-%H-%M')+'.png')
    plt.figure()
    plt.plot(scores)
    plt.xlabel("time")
    plt.ylabel('score')
    plt.savefig('savefig'+datetime.datetime.now().strftime('%d-%H-%M')+'2.png')




def main():
    trainNetwork(istrain = True)

if __name__ == "__main__":
    main()
