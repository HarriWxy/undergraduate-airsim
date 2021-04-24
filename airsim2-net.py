# !/usr/bin/env python
#============================ 导入所需的库 ===========================================
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense,Dropout

# import cv2
import sys
import random
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt
import datetime

from tornado.util import xrange 
os.environ['CUDA_VISIBLE_DEVICES']='0'

import get_stategcopy as state

ACTIONS = 6 # 4个动作数量
ACTIONS_NAME=['forward','back','roll_right','roll_left','higher','lower','yaw_left','yaw_right']  #动作名
GAMMA = 0.99 # 未来奖励的衰减
OBSERVE = 32. # 训练前观察积累的轮数
EPSILON = 0.98
REPLAY_MEMORY = 1000 # 观测存储器D的容量
BATCH = 32 # 训练batch大小
old_time = 0

class DQN_Net(Model):
    # 使用论文中的标准网络结构
    def __init__(self):
        super(DQN_Net, self).__init__()
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

        # self.c3_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
        #                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
        #                    bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        # #self.b1 = BatchNormalization()  # BN层
        # self.a3_1 = Activation('relu')  # 激活层
        # self.c3_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
        #                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
        #                    bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        # #self.b1 = BatchNormalization()  # BN层
        # self.a3_2 = Activation('relu')  # 激活层
        # self.c3_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
        #                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
        #                    bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        # #self.b1 = BatchNormalization()  # BN层
        # self.a3_3 = Activation('relu')  # 激活层
        # self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))
        self.f2 = Dense(ACTIONS, activation=None,
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))

    def call(self, x):
        # print(x.shape)
        # # t=tf.Variable(0.)
        # # for j in x :
        # #     t+= self.c1_1(j)
        # # x=t
        # print(x.shape)
        x = self.c1_1(x)
        x = self.a1_1(x)
        x = self.d1(x)
        x = self.p1(x)

        x = self.c2_1(x)
        x = self.a2_1(x)
        x = self.c2_2(x)
        x = self.a2_2(x)
        x = self.p2(x)

        # x = self.c3_1(x)
        # x = self.a3_1(x)
        # x = self.c3_2(x)
        # x = self.a3_2(x)
        # x = self.c3_3(x)
        # x = self.a3_3(x)
        # x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        y = self.f2(x)
        # print(y)
        return y

class AirsimDQN(object):
    def __init__(self):
        super().__init__()
        self.net = DQN_Net()
        self.epco = 0        

    @tf.function #暂时不可用
    def train_step(self,trans):
        # 训练函数
        print("trsiandsbnjk")
        [b_s, b_a, b_r, b_s_, b_done]=trans
        with tf.GradientTape() as tape:
            q= self.net(b_s)
            q_next=self.net(b_s_)
            q_truth = b_r + GAMMA * q_next* (1 - b_done)
            loss = tf.losses.MSE(q_truth, q)
            print("loss:",float(loss))
            # print("loss = %.4f" % float(loss))
            gradients = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

    def trainNetwork(self,istrain):
        self.epco += 1
    #============================ 模型创建与加载 ===========================================

        # 模型创建
        
    #============================ 配置模型 ===========================================
        # optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6, epsilon=1e-08)  #1e-6
        self.optimizer=tf.keras.optimizers.RMSprop(0.00025,0.99,0.0,1e-7)
        # 学习率等参数后期调节
        t = 0 #初始化TIMESTEP
        losses=[]
        # 加载保存的网络参数
        checkpoint_save_path = "./model/Airsim"
        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            self.net.load_weights(checkpoint_save_path)
        else:
            print('-------------train new model-----------------')

    #============================ 加载(搜集)数据集 ===========================================

        # 打开飞行模拟
        # flying_state = state.FlyingState(random.randint(-40,40),random.randint(-40,40),random.randint(-40,0))
        flying_state = state.FlyingState(20,20,-40)
        # destination:(x,y,z)

        # 将每一轮的观测存在队列D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
        D = deque()

        #初始化状态并且预处理图片，把连续的四帧图像作为一个输入(state)
        do_forward = np.zeros(ACTIONS)
        # 在action lists中选中对应的操作设置为1
        do_forward[0] = 1
        x_t, r_0, terminal, _ = flying_state.frame_step(do_forward)
        # x_t=cv2.resize(x_t,(84,84,4))# 84*84*4作为输入
        # x_t = cv2.cvtColor(x_t, cv2.COLOR_BGR2GRAY)
        # 这里有点疑惑是为啥哈
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
        netstar=DQN_Net()
        best_checkpoint_save_path = "./best/AirSim"
        if os.path.exists(best_checkpoint_save_path + '.index'):
            netstar.load_weights(best_checkpoint_save_path)
        else:
            netstar.set_weights(self.net.get_weights())
        
        # 开始训练
        while t < 1000:
            epsilon = EPSILON - ((1.0-0.1)/1000)*t
            # 根据输入的s_t,选择一个动作a_t
            
            # 图像信息作为直接输入
            a_t_to_drone = np.zeros([ACTIONS])
            action_index = 0

            #贪婪策略，有episilon的几率随机选择动作去探索，否则选取Q值最大的动作
            if random.random() <= epsilon and istrain:
                print("----------Random Action----------")
                dirction = random.randint(0,3)
                action_index = flying_state.rand_action(dirction)
                a_t_to_drone[action_index] = 1
            else:
                print("-----------net choice----------------")
                readout_t = self.net(tf.constant(s_t, dtype=tf.float32))
                action_index = np.argmax(readout_t[-1])
                # 全连接层中概率最大的方向
                print("-----------index----------------")
                print(action_index)
                a_t_to_drone[action_index] = 1

            #执行这个动作并观察下一个状态以及reward
            x_t1, r_t, terminal, score = flying_state.frame_step(a_t_to_drone)
            print("============== score ====================")
            print(score)

            rank_file_r = open("rank.txt","r")
            # 最优得分
            best = int(rank_file_r.readline())
            rank_file_r.close()
            #if score_one_round >= best:
            #    test = True
            if score > best:
                self.net.save_weights(best_checkpoint_save_path)
                netstar.set_weights(self.net.get_weights())
                rank_file_w = open("rank.txt","w")
                rank_file_w.write("%d" % score)
                print("********** best score updated!! *********")
                rank_file_w.close()
            if score >= best:
                f = open("scores.txt","a")
                f.write("========= %d ========== %d \n" % (t+old_time, score))
                f.close()

            a_t = action_index # action
            # x_t1 = cv2.cvtColor(x_t1, cv2.COLOR_BGR2GRAY)
            # ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            # luminance=cv2.cvtColor(x_t1, cv2.COLOR_BGR2HSV)
            # x_t1=x_t1[:,luminance[0]]# 添加亮度信息
            # x_t1=cv2.resize(x_t,(84,84,4))# 84*84*4作为输入
            s_t1=np.array([s_t[1,:,:],s_t[2,:,:],s_t[3,:,:],x_t1],dtype=np.int)
            # print(x_t1.shape)
            # s_t1 = np.append(s_t[:3],x_t1, axis=0)
            # print(s_t1.shape)

            s_t_D = tf.convert_to_tensor(s_t, dtype=tf.float32)
            a_t_D = tf.constant(a_t, dtype=tf.int32)
            r_t_D = tf.constant(r_t, dtype=tf.float32)
            s_t1_D = tf.convert_to_tensor(s_t1, dtype=tf.float32)
            terminal = tf.constant(terminal, dtype=tf.float32)

            # 将观测值存入之前定义的观测存储器D中
            D.append((s_t_D, a_t_D, r_t_D, s_t1_D, terminal))
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
                # print(minibatch)

                # 训练
                with tf.GradientTape() as tape:
                    # 获得batch中的每一个变量
                    loss_even=0.0
                    for d in minibatch:
                    # self.train_step(d) #这个用不起需要后续研究一下
                        [b_s, b_a, b_r, b_s_, b_done]=d
                        # b_s=tf.convert_to_tensor(b_s, dtype=tf.float32)
                        # b_r= tf.constant(b_r, dtype=tf.float32)
                        # b_s_ = tf.convert_to_tensor(b_s_, dtype=tf.float32)
                        # b_done = tf.constant(b_done, dtype=tf.float32)
                        q = tf.reduce_max(self.net(b_s)[-1])
                        q_next=tf.reduce_max(netstar(b_s_)[-1])
                        q_truth = b_r + GAMMA * q_next* (1 - b_done)
                        # loss = tf.losses.MSE(q_truth, q)
                        loss=(q_truth-q)**2
                        # print("loss:",float(loss))
                        loss_even+=loss
                        # print("loss = %.4f" % float(loss))
                    loss_even=loss_even/BATCH
                    losses.append(loss_even)
                    print("loss = %.4f" % float(loss_even))
                    gradients = tape.gradient(loss_even, self.net.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

                # if (t+old_time) % 20 == 0:
                #     netstar=self.net

                # 每100轮保存一次网络参数
                if (t+old_time) % 100 == 0:
                    print("=================model save====================")
                    self.net.save_weights(checkpoint_save_path)

            # 打印信息

            print("TIMESTEP", (t+old_time), "|  ACTION", ACTIONS_NAME[action_index], "|  REWARD", r_t)#, \
            # "|  Q_MAX %e \n" % np.max(readout_t))
            # write info to files
        plt.plot(losses)
        plt.xlabel("time")
        plt.ylabel('loss')
        plt.savefig('savefig'+datetime.datetime.now().strftime('%d-%H-%M')+str(self.epco)+'.png')



def main():
    air_DQN=AirsimDQN()
    # while True:
    air_DQN.trainNetwork(istrain = True)

if __name__ == "__main__":
    main()