import numpy as np
import airsim
import cv2 as cv
from numpy.core.fromnumeric import reshape, shape

# 获取airsim飞行状态
class FlyingState:
    def __init__(self,x,y,z):
        self.dest=(x,y,z)
        self.score = 0
        self.client=airsim.MultirotorClient()
        self.linkToAirsim()
        # self.client.moveToPositionAsync()自动移动到目的地
        # client take off


    def linkToAirsim(self):
        # 连接到airsim
        self.client.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # connect to the AirSim simulator 
        self.client.takeoffAsync().join()

    def frame_step(self, input_actions):
        # 执行操作并获取帧
        reward = 0
        terminal = False
        client_state=self.client.getMultirotorState()
        client_pre_pos=client_state.kinematics_estimated.position

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # ['forward','back','roll_right','roll_left','yaw_left','yaw_right','higher','lower']
        if input_actions[0]==1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.2,0.0,0.6,0.5).join()
        elif input_actions[1] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,-0.2,0.0,0.6,0.5).join()
        elif input_actions[2] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.2,0.0,0.0,0.6,0.5).join()
        elif input_actions[3] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(-0.2,0.0,0.0,0.6,0.5).join()
        elif input_actions[4] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.0,0.0,1.0,0.5).join()
        elif input_actions[5] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.0,0.0,0.5,0.5).join()
        elif input_actions[6] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.0,15.0,0.8,0.5).join()
        elif input_actions[7] == 1:
            self.client.moveByRollPitchYawrateThrottleAsync(0.0,0.0,-15.0,0.8,0.5).join()

        # client state
        client_state=self.client.getMultirotorState().kinematics_estimated
        client_position=client_state.position
        # position and crash
        Crash_info=self.client.simGetCollisionInfo().has_collided
        # rewards:closer +1 or farther -1
        dis_pre=np.linalg.norm([self.dest[0]-client_pre_pos.x_val,self.dest[1]-client_pre_pos.y_val,self.dest[2]-client_pre_pos.z_val])
        dis_this=np.linalg.norm([self.dest[0]-client_position.x_val,self.dest[1]-client_position.y_val,self.dest[2]-client_position.z_val])
        if dis_this < dis_pre:
            reward=1
            self.score+=reward
        else:
            reward=-1
            self.score+=reward
        # front camera scene
        response=self.client.simGetImages([airsim.ImageRequest("0",airsim.ImageType.Scene,False,False)])
        response=response[0]
        image_data=np.frombuffer(response.image_data_uint8,dtype=np.uint8)
        image_data=image_data.reshape(response.height,response.width,3)
        # print(image_data)
        # image_data=np.reshape(image_data,[response.height,response.width,3])
        image_data=np.flipud(image_data)
        # image_data=image_data[:,:]
        # # [height,width,channel]=image_data.shape
        ret, image_data = cv.threshold(image_data, 1, 255, cv.THRESH_BINARY)
        luminance=cv.cvtColor(image_data, cv.COLOR_BGR2HSV)
        # # image_data=image_data.astype(np.float)
        image_data=np.array([image_data[:,:,0],image_data[:,:,1],image_data[:,:,2],luminance[:,:,2]],dtype=np.int)# 添加亮度信息
        # image_temp=np.array([])
        # image_data=cv.cvtColor(image_data,cv.COLOR_BGR2GRAY)
        # print(image_data.shape)
        score=self.score
        if Crash_info or self.score < -10:
            self.score=0
            terminal=True
            self.linkToAirsim()
            reward=-1
            score+=reward
        if dis_this < 1:
            self.score=0
            reward = 100
            terminal=True
            self.linkToAirsim()
            score+=reward
        return image_data, reward, terminal, score

    def rand_action(self,randint):
        client_state=self.client.getMultirotorState().kinematics_estimated
        if randint == 1:
            return 0 if self.dest[0] > client_state.position.x_val else 1
        elif randint == 2:
            return 2 if self.dest[1] > client_state.position.y_val else 3
        else :
            return 4 if self.dest[2] < client_state.position.z_val else 5



# responses=client.simGetImages([
#     airsim.ImageRequest(0,airsim.ImageType.Scene),
#     # 前视深度信息
#     airsim.ImageRequest(0,airsim.ImageType.DepthVis),
#     # bottom深度信息
#     airsim.ImageRequest(3,airsim.ImageType.DepthVis)
# ])
if __name__ == "__main__":
    game_state = FlyingState(200,200,200)
    for i in range(0,20000):
        a_t_to_game = np.zeros(8)
        # action_index = random.randrange(2)
        a_t_to_game[0] = 1
        game_state.frame_step(a_t_to_game)
        game_state.rand_action(2)