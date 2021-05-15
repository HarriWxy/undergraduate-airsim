# !/usr/bin/env python
#============================ 导入所需的库 ===========================================
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, LSTM

# import cv2
import sys
import random
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt
import datetime

os.environ['CUDA_VISIBLE_DEVICES']='0'

import get_state as state

ACTIONS = 6 # 4个动作数量
ACTIONS_NAME=['forward','back','roll_right','roll_left','higher','lower','yaw_left','yaw_right']  #动作名
GAMMA = 0.9 # 未来奖励的衰减
OBSERVE = 10 # 训练前观察积累的轮数
EPSILON = 0.15
REPLAY_MEMORY = 250# 观测存储器D的容量
BATCH = 7 # 训练batch大小
TIMES = 5000

class DQN_Net(Model):
    # 使用论文中的标准网络结构
    def __init__(self):
        super().__init__()
        self.c1_1 = Conv2D(filters=32, kernel_size=(8, 8),strides=4, padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        self.a1_1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.1)  # dropout层

        self.c2_1 = Conv2D(filters=64, kernel_size=(4, 4),strides=2, padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        self.a2_1 = Activation('relu')  # 激活层
        self.c2_2 = Conv2D(filters=64, kernel_size=(3, 3),strides=1, padding='same',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        self.a2_2 = Activation('relu')  # 激活层
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
        self.l1 = LSTM(256,dropout=0.1,kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.f2 = Dense(ACTIONS, activation=None,
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))

    def call(self, x, directions):
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
        x = tf.expand_dims(x,axis=0)
        x = self.l1(x)
        directions = tf.expand_dims(directions,axis=0)
        x = tf.concat((x,directions),axis=1)
        y = self.f2(x)
        print("y=",y)
        return y

class AirsimDQN(object):
    def __init__(self):
        super().__init__()
        #============================ 模型创建与加载 ===========================================
        self.net = DQN_Net()
        self.netstar = DQN_Net()
        self.checkpoint_save_path = "./model/AirSim"
        if os.path.exists(self.checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            self.net.load_weights(self.checkpoint_save_path)
        else:
            print('-------------train new model-----------------')
        self.best_checkpoint_save_path = "./best/AirSim"
        if os.path.exists(self.best_checkpoint_save_path + '.index'):
            self.netstar.load_weights(self.best_checkpoint_save_path)
        # 最高分的加载
        rank_file_r = open("rank.txt","r")
        self.best = int(rank_file_r.readline())
        rank_file_r.close()
        self.epoch = 0     
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        # self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)   

    def trainNetwork(self,istrain):
        self.epoch+=1
    #============================ 配置模型 ===========================================
        # optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-6, epsilon=1e-08)  #1e-6

        t = 0 #初始化TIMESTEP
        losses=[]
        scores=[]
        Loss=0
        qs=[]
        # Score=0

        # 加载保存的网络参数
       
    #============================ 加载(搜集)数据集 ===========================================

        # 打开游戏
        flying_state = state.FlyingState(random.randint(-30,30),random.randint(-30,30),-30)

        # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
        D = deque()

        #初始化状态并且预处理图片，把连续的四帧图像作为一个输入（State）
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal, _, directions = flying_state.frame_step(do_nothing)
        s_t = np.stack((x_t,x_t,x_t,x_t,x_t),axis=0)
        directions = tf.convert_to_tensor(directions,dtype=tf.float32)

        # 开始训练
        while t < TIMES+1:
            # 根据输入的s_t,选择一个动作a_t
            # 网络的过早介入会导致
            # 学习率
            if t < 1500 :
                epsilon = EPSILON - (EPSILON-0.1)*t/1500
                # learning_r= 0.005# - (0.001-0.00025)*t/450
            else :
                epsilon = 0.1
            learning_r=0.00075
            optimizer=tf.keras.optimizers.RMSprop(learning_r,0.99,0.0,1e-7)
            
            a_t_to_game = np.zeros([ACTIONS])
            action_index = 0

            #贪婪策略，有episilon的几率随机选择动作去探索，否则选取Q值最大的动作
            if random.random() <= epsilon and istrain:
                print("----------Random Action----------")
                dirction = random.randint(0,3)
                # action_index = flying_state.rand_action(dirction)
                action_index = random.randint(0,5)
                a_t_to_game[action_index] = 1
            else:
                print("-----------net choice----------------")
                readout_t = self.net(tf.constant(s_t, dtype=tf.float32),directions)
                action_index = np.argmax(readout_t)
                print("-----------index----------------")
                print(action_index)
                a_t_to_game[action_index] = 1

            #执行这个动作并观察下一个状态以及reward
            x_t, r_t, terminal, score, directions = flying_state.frame_step(a_t_to_game)
            x_t_n = x_t[np.newaxis,:]
            s_t = np.concatenate((s_t[1:],x_t_n))
            print("============== score ====================")
            print(score)
            # if t > 1800 :
            #     scores.append(r_t)
            #     qs.append(int(q))
            #if score_one_round >= best:
            #    test = True

            # if score >= best:
            #     f = open("scores.txt","a")
            #     f.write("========= %d ========== %d \n" % (t+old_time, score))
            #     f.close()

            # a_t = action_index

            s_t_D = tf.convert_to_tensor(s_t, dtype=tf.float32) # 下一帧
            action_index_D = tf.constant(action_index, dtype=tf.int32)
            r_t = tf.constant(r_t, dtype=tf.float32)
            terminal_D = tf.constant(terminal, dtype=tf.float32)
            directions = tf.convert_to_tensor(directions,dtype=tf.float32)

            # 如果回合结束下一个状态重新开始
            if terminal :
                s_t = np.stack((x_t,x_t,x_t,x_t,x_t),axis=0)
            # 将观测值存入之前定义的观测存储器D中
            D.append((s_t_D, r_t, terminal_D, action_index_D, directions))
            #如果D满了就替换最早的观测
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # 更新状态，不断迭代
            t += 1

    #============================ 训练网络 ===========================================

            # 观测一定轮数后开始训练
            if (t > OBSERVE) and istrain:
                if score > self.best:
                    self.net.save_weights(self.best_checkpoint_save_path)
                    rank_file_w = open("rank.txt","w")
                    rank_file_w.write("%d" % score)
                    rank_file_w.close()
                    print("********** best score updated!! *********")
                    self.best = score
                    self.netstar.set_weights(self.net.get_weights())
                # 训练
                print("==================start train====================")
                with tf.GradientTape() as tape:
                    q = self.net(s_t_D[:-1],directions)[0][action_index]
                    q_next = self.netstar(s_t_D[1:],directions)[0][action_index] # 下一个状态的Y函数值
                    q_truth = r_t + GAMMA * q_next* (tf.ones(1) - terminal) # 
                    # print("q=",q,q_truth)
                    loss = tf.losses.MSE(q_truth, q)
                    

                    # minibatch
                    for i in random.sample(D, BATCH-1): 
                        b_s = i[0]
                        b_r = i[1]
                        b_done = i[2]
                        b_a = int(i[3])
                        b_d = i[4]
                        q = self.net(b_s[:-1],b_d)[0][b_a]
                        q_next = self.netstar(b_s[1:],b_d)[0][b_a] # 每一行出一个最大值
                        if int(q_next) < -10:
                            q_next = tf.constant(-10,dtype=tf.float32)
                        elif int(q_next) > 50:
                            q_next = tf.constant(50,dtype=tf.float32)
                        q_truth = b_r + GAMMA * q_next* (tf.ones(1) - b_done)
                        loss += tf.losses.MSE(q_truth, q)

                    # # 训练
                    loss = loss/BATCH
                    Loss += loss
                    print("loss = %f" % loss)
                    gradients = tape.gradient(loss, self.net.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

                if t % 30==0 and t > OBSERVE:
                    self.netstar.set_weights(self.net.get_weights())

                # 每1000轮保存一次网络参数
                if t % 50 == 0:
                    # print("=================model save====================")
                    self.net.save_weights(self.checkpoint_save_path)
                    Loss = Loss/50

                    # 这是使用tensorboard
                    # with self.train_summary_writer.as_default():
                    #     losses = np.array(losses)
                    #     scores = np.array(scores)
                    #     tf.summary.scalar('loss',Loss,self.epoch)
                    #     tf.summary.scalar('score',Score,self.epoch)

                    # 在此调用测试函数评估得分

                    losses.append(Loss)
                    # scores.append(Score)
                    # Score=0
                    Loss = 0

            # 打印信息

            print("TIMESTEP", t, "|  ACTION", ACTIONS_NAME[action_index], "|  REWARD", r_t)#, \
            #    "|  Q_MAX %e \n" % np.max(readout_t))
            # write info to files
        
        plt.figure()
        plt.title("epochs="+str(self.epoch))
        plt.plot(losses)
        plt.xlabel("time")
        plt.ylabel('loss')
        plt.savefig('savefig'+datetime.datetime.now().strftime('%d-%H-%M')+'.png')
        # plt.figure()
        # plt.title("epochs="+str(self.epoch))
        # plt.plot(scores,'b')
        # plt.xlabel("time")
        # plt.ylabel('score')
        # plt.plot(qs,'c--')
        # plt.savefig('savefig'+datetime.datetime.now().strftime('%d-%H-%M')+'2.png')



def main():
    air_DQN=AirsimDQN()
    while True:
        air_DQN.trainNetwork(istrain = True)

if __name__ == "__main__":
    main()
