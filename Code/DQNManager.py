import pickle
import random
from DQN import DQN ,DQN_v1
import collections
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
from enum import Enum
import os
import Config
import Config
from torch.utils.tensorboard import SummaryWriter


class ModelMode(Enum):
    INFERENCE = 1
    TRAIN = 2


class PredictMode(Enum):
    FINAL = 1
    BEST = 2 
    SELECT = 3

class DQN_Manager():
    '''
    초기화 함수 
    _epi : _predictMode가 SELECT일 경우 모델 Load시 필요한 epi 정보
    '''
    def __init__(self, _inPut:int, _outPut:int,_modelMode:ModelMode, _predictMode:PredictMode,_epi):
        self.gamma = Config.gamma 
        self.buffer_limit = Config.buffer_limit
        self.batch_size = Config.batch_size 
        self.learning_rate = Config.learning_rate
        self._modelMode = _modelMode
        self._predictMode = _predictMode

        if self._modelMode == ModelMode.TRAIN:
            if not os.path.exists(Config.tensorboardLog):
                os.makedirs(Config.tensorboardLog) 
            self.tensorWriter = SummaryWriter(Config.tensorboardLog)

        #self.best_avg_reward = -np.inf 
        #self.cur_reward = [] # best 모델 선정 비교에서 사용
        self.step_done = 0 # 진행된 모델 학습 수 
        self.updatePeriod = Config.updatePeriod # TargetNetwork 업데이트 주기

        # 모델 선언
        self.QNetwork = DQN_v1(_inPut,_outPut)
        self.TargetNetwork = DQN_v1(_inPut,_outPut)
        # 옵티마이저 선언 
        self.optimizer = optim.Adam(self.QNetwork.parameters(), lr = self.learning_rate)

        self.memory = collections.deque(maxlen=self.buffer_limit)
        self.LoadModel(_epi)

        self.TargetNetwork.load_state_dict(self.QNetwork.state_dict())


    def __del__(self):
        pass

    '''
    epsilon에 따라서 랜덤행동을 선택할지 모델이 예측한 행동을 선택할지 리턴해주는 함수
    '''
    def Select_action(self,status:np.ndarray,epsilon:float,actionCount:int,_modelMode:ModelMode):
        coin = random.randint(0,100)/100
        
        status = status.reshape(1,-1) # 모델의 input에 맞게 shape 변경
        with torch.no_grad():
            self.QNetwork.eval() # 예측할 경우 배치노멀라이제이션을 비활성화
            actionTensor = self.QNetwork(torch.from_numpy(status).float())
        self.QNetwork.train() # 다시 학습해야 하므로 배치노멀라이제이션 활성화 

        if _modelMode == ModelMode.TRAIN:
            if coin < epsilon:
                return random.randint(0,actionCount), coin
            else:
                return actionTensor.argmax().item(), coin # 큐값 높은 인덱스 리턴 
        else:
            with torch.no_grad():  
                return actionTensor.argmax().item(), coin 
    
    '''
    Memory에 모델을 학습 시킬 데이터를 삽입하는 함수 
    ''' 
    def PushMemory(self,state,action,reward,next_state,done):
        #self.cur_reward.append(reward)
        self.memory.append([state,action,reward,next_state,done])
    
    '''
    모델을 학습 시킬 Sample 데이터를 Batch Size만큼 추출하는 함수
    '''
    def GetSample(self):
        try:
            mini_batch = random.sample(self.memory, self.batch_size)
            s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
            
            for transition in mini_batch:
                s, a, r, s_prime, done_mask = transition
                
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                done_mask_lst.append([done_mask])

            return torch.tensor(np.array(s_lst), dtype=torch.float), \
                    torch.tensor(np.array(a_lst)), torch.tensor(np.array(r_lst)), \
                    torch.tensor(np.array(s_prime_lst), dtype=torch.float), torch.tensor(np.array(done_mask_lst))
        except Exception as ex:
            print("에러발생", ex)
            
    '''
    DQN의 Memory 크기 반환
    '''
    def MemorySize(self):
        return len(self.memory)
    
    '''
    모델 학습 함수
    - Memory에 train_size 만큼 쌓은 수 부터 학습 시작
    - 
    '''
    def TrainModel(self):
        self.step_done += 1
        
        if self.MemorySize() < Config.train_size:
            return
        
        states, actions, rewards, next_states, dones = self.GetSample()

        q_out = self.QNetwork(states)
        q_a = q_out.gather(1, torch.tensor(np.array(actions))) # q_a : 모델이 모든 상태에 대해 예측한 것중 실제 행동한 actions 값으로 가치함수 추출
        max_q_prime = self.TargetNetwork(next_states).max(1)[0].unsqueeze(1) # 모델이 다음 상태를 예측한 값에서 가치함수가 최대인 것을 추출 후 unsqueeze()를 통해
                                                                             # q_a와 차원을 동일하게 맞춘다. 파라미터 위치(1)에 차원 추가
        target = rewards + self.gamma * max_q_prime * dones # DQN 공식에 따라서 target값 정의

        loss = F.smooth_l1_loss(q_a, target)

        # Optimize the model
        self.tensorWriter.add_scalar("Loss/train",loss,self.step_done) # TensorBoard에 Loss값 기록 
        self.optimizer.zero_grad()  # gradient 초기화
        loss.backward()             # gradient 계산
        self.optimizer.step()       # 계산된 gradient로 모델 파라미터 업데이트
        self.UpdateTargetNetwork()

    '''
    TargetNetwork를 업데이트 하는 함수
    - step_done의 배수 마다 업데이트 
    '''
    def UpdateTargetNetwork(self):
        if self.step_done % self.updatePeriod == 0 and self.step_done != 0:
            print("TargetNetwork update : ", self.step_done)
            self.TargetNetwork.load_state_dict(self.QNetwork.state_dict())

    # def UpdateBestAvgReward(self):
    #     cur_avg_reward = np.mean(self.cur_reward)
    #     if cur_avg_reward > self.best_avg_reward:
    #         self.best_avg_reward = cur_avg_reward
    #     self.cur_reward.clear()

    '''
    _pMode에 따라 모델 이름을 저장하는 방식 변환
    _epi는 _pMode가 SELECT일때만 사용 
    .tar : Model 정보에는 가중치와 옵티마이저 값 저장 
    .dump : 모델 학습 중 쌓인 Memory 저장
    '''
    def Savemodel(self, _pMode:PredictMode, _epi=0):
        if self._modelMode == ModelMode.TRAIN:
            if _pMode == PredictMode.FINAL:
                _modelName = Config._finalModelName 
            elif _pMode == PredictMode.BEST: 
                _modelName = Config._bestModelName
            else :
                _modelName = 'epi' + str(_epi) + Config._selectModelName    
            _modelPath = os.path.join(Config._modelTrainPath, f'{_modelName}.tar')
            torch.save({'model_state' : self.QNetwork.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}, 
                        #'best_reward' : self.best_avg_reward},
                    _modelPath)

            if _pMode != PredictMode.SELECT:
                _mDumpPath = os.path.join(Config._modelTrainPath, f'{_modelName}.dump')
                with open(_mDumpPath, "wb") as _mFile:
                    pickle.dump(self.memory, _mFile)

    '''
    저장된 모델을 읽을때 사용하는 함수
    - 재학습 시 저장된 Memory Load
    - 저장된 모델 가중치, 옵티마이저 값 적용 
    '''
    def LoadModel(self,_epi:int):
        try:
            if self._predictMode == PredictMode.FINAL:
                _modelName = Config._finalModelName 
            elif self._predictMode == PredictMode.BEST: 
                _modelName = Config._bestModelName
            else :
                _modelName = 'epi' + str(_epi) + Config._selectModelName
                
            if self._modelMode == ModelMode.TRAIN and self._predictMode != PredictMode.SELECT:
                _mDumpPath = os.path.join(Config._modelTrainPath, f'{_modelName}.dump')
                if(os.path.isfile(_mDumpPath)):
                    with open(_mDumpPath, "rb") as _mFile:
                        self.memory = pickle.load(_mFile)

            _modelPath = os.path.join(Config._modelTrainPath, f'{_modelName}.tar')
            if(os.path.isfile(_modelPath)):
                _checkpoint = torch.load(_modelPath) # _checkpoint Type : dict
                self.QNetwork.load_state_dict(_checkpoint['model_state'])
                self.optimizer.load_state_dict(_checkpoint['optimizer'])
                #self.best_avg_reward = _checkpoint['best_reward']
                if self._modelMode == ModelMode.INFERENCE:
                    self.QNetwork.eval() # 드롭아웃, 배치노멀라이제이션 사용 x
                else:
                    self.QNetwork.train() # 드롭아웃, 배치노멀라이제이션 사용 o
        except Exception as ex:
            print("에러발생", ex)