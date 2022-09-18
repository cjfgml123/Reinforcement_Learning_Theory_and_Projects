import traceback
from DQNManager import DQN_Manager, ModelMode, PredictMode
from TLS import TOD, Pattern, PlanType
import numpy as np
import Config
from TLSManager import TLSManager
from Env_Sumo import RewardType, SumoSimulator, SimulMode as sMode, TODMode
import Env_Sumo
import SumoUtil
import pandas as pd
from Util import CSVManager as csvMgr
import datetime
from TrafficDataManager import TraffciDataManager
from DataManager import DataManager
from torch.utils.tensorboard import SummaryWriter

# 현시 행동 목록
'''
모델 학습 및 예측 프로세스
'''
def RunRL(_modelMode = ModelMode.TRAIN,
          _predictMode = PredictMode.FINAL,
          _todMode = TODMode.Manual,
          _simulMode=sMode.Fixed, 
          _planType:PlanType=PlanType.WeekDay,
          _patternIndex:int=0,
          _modelIDepi:int=None):
    dqnManager = None # DQNManager 객체
    _epsilon = 0 # DQN epsilon 값
    _score = 0 # 하루 누적 Reward 값
    _accumulatePassTime = 0 # 하루 주기별 누적 통과시간 
    best_score = -np.inf # 학습 과정 중 최고 Reward 값 (학습 데이터 내에서)
    _eps_start = Config._eps_start # DQN epsilon 시작값
    _data = None # vds 데이터 객체
    _intersectionState = "" # 강화학습 State 값
    _valPassTime = np.inf # 모델 학습 검증 과정 중 하루 최소 통과시간
    _valReward = -np.inf # 모델 학습 검증 과정 중 하루 최대 reward
    _beforeReward = 0 # case11은 전주기reward 와 다음 주기 reward 차이값을 이용할때 쓰이는 변수 (case11 Reward : 최대 밀도)
    _historyState = 'SUCCESS' # DB에 이력 기록할때 에러 시 'Fail' 기록
    _errorMsg = ''  # 에러 메시지
    _tdMgr = None # DBManager 객체
    try:
        # TOD DB 생성
        _tlsMgr = TLSManager()
        _newTOD: TOD = _tlsMgr.GetTOD(Config.NODE_ID, Config.NodeName)
        _actionCount = _newTOD.GetPatternCount()
        _tdMgr = TraffciDataManager(Config._data_DBName, Config._ip, Config._port, Config._id, Config._pw)
        # Source 데이터 로드
        if _modelMode == ModelMode.TRAIN:
            _data = pd.read_csv(Config._vdsOriginFilePath) # 학습용 데이터 로드
            _valData = pd.read_csv(Config._vdsValFilePath) # 검증용 데이터 로드 
        else:
            _data = pd.read_csv(Config._vdsFilePath) # 예측 데이터
            
        _dataMgr = DataManager()
        _dataMgr.Initialize()
        _intersectionID = _dataMgr.GetIntersectionID(Config.NODE_ID)
            
        # 시뮬레이션 인스턴스 생성
        sumo_sim = SumoSimulator(_todMode=_todMode, _sMode=_simulMode, _tod=_newTOD, 
                                 _planType=_planType, _patternIndex=_patternIndex, _rewardType=RewardType.DensityAndPassTime)
        
        _state = [0 for i in(range(Config.modelInputSize))] # 모델 Input 사이즈를 갖는 List생성

        _modelID = _tdMgr.SelectModel(Config._finalModelName)
        
        #새로운 모델을 학습할 경우
        if _modelID is None: 
            _modelID = _tdMgr.InsertModel(Config._finalModelName,len(_state), _actionCount, Config.buffer_limit,
                                          Config.learning_rate,Config.batch_size,
                                          Config.updatePeriod,Config.gamma,Config.train_size,
                                        Config._eps_start,Config._eps_end,Config._episodes)
        else : # 모델 재학습 (학습 이어서 모델 생성)
            _tdMgr.UpdateModel(_modelID,len(_state), _actionCount, Config.buffer_limit,
                                          Config.learning_rate,Config.batch_size,
                                          Config.updatePeriod,Config.gamma,Config.train_size,
                                        Config._eps_start,Config._eps_end,Config._episodes)
        
        _data[['DAY', 'TIME']] = _data['TOT_DT'].str.split(' ',n=1, expand=True)
        _vdsDayGroup = _data.groupby(['DAY'])
        
        _groupKeyList = list(_vdsDayGroup.groups.keys())
        _gruopIndex = 0
        _eCount = 1

        _start_epi = 0
        if _modelMode == ModelMode.TRAIN:
            _historyID = _tdMgr.SelectHistory(_modelID)
            if _historyID is not None: # 모델 재학습 시 시작epi , best모델 선정 기준인 검증 데이터에서 하루 동안의 차량당 통과시간 및 하루 누적 Reward 값 조회
                _start_epi,_valPassTime,_valReward = _tdMgr.SelectProgressedVals(_historyID) 
                if _start_epi is not None:
                    _start_epi += 1
                else :
                    _start_epi = 0
                if _valPassTime is None:
                    _valPassTime = np.inf
                if _valReward is None:
                    _valReward = -np.inf

            _gruopIndex = _start_epi % len(_groupKeyList)
            _eCount = Config._episodes
                
            _dateTimeNowStr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            _historyID = _tdMgr.InsertHistory(_modelID,_dateTimeNowStr)
        else :
           _predictID = _tdMgr.InsertPredictInfo(_modelID, _predictMode.name, _groupKeyList[0])
           

        dqnManager = DQN_Manager(len(_state), _actionCount,_modelMode,_predictMode,_modelIDepi)
        
        for epi in range(_start_epi, _eCount):
            _beforeReward = {}
            _beforeReward['충청대학교 방향'] = 0
            _beforeReward['오송역 방향'] = 0
            _beforeReward['청주역 방향'] = 0
            _beforeReward['청주IC 방향'] = 0
            dateTimeStr = _groupKeyList[_gruopIndex]
            _timeGroup = _vdsDayGroup.get_group(dateTimeStr).groupby(['TIME'])
            
            # 훈련용 데이터에서 누락 데이터 개수(15분단위 기준)가 있으면 그 날짜의 vds 데이터는 사용 X
            if len(_timeGroup) != 93:
                _gruopIndex += 1
                if _gruopIndex >= len(_groupKeyList):
                    _gruopIndex = 0
                continue
            
            if not SumoUtil.CreateTripFile(_timeGroup):
                print('Trip 파일 생성 실패')

            _target = len(_timeGroup) * 900
            sumo_sim.Start(_groupKeyList[_gruopIndex], _target)

            _gruopIndex += 1
            if _gruopIndex >= len(_groupKeyList):
                _gruopIndex = 0

            _state = np.array(_state)
            
            # _eps_start 부터 _eps_end가 될때까지 학습 하면서 _epsilon 값을 점차 줄임.
            _epsilon = max(Config._eps_end, _eps_start - Config._eps_end*(epi/1.7))
            
            if _modelMode == ModelMode.TRAIN:
                _tdMgr.EveryUpdateHistory(_historyID,epi,round(_epsilon,3))
                _epiID = _tdMgr.InsertEpisode(_historyID,epi,_epsilon)
            
            _datetime = datetime.datetime.strptime(dateTimeStr, '%Y-%m-%d')
            _day = _datetime.date()
            nationalDayList = Env_Sumo.GetNationalDay()
            if _day in nationalDayList: # 국경일 여부 (state에 들어감)
                _isNationalDay = 1
            else :
                _isNationalDay = 0
            
            for _currentTime in sumo_sim:
                _episodeState = 'SUCCESS'
                try:
                    _nextStates = []
                    _statesRawDataList = []
                    _action_Index, _percent = dqnManager.Select_action(_state, _epsilon, _actionCount-1,_modelMode)
                    # 시뮬레이션 진행
                    for i in range(Config.LOOP_COUNT):
                        _simulResult, _isDone = sumo_sim.RunCycleSimulation(_action_Index)
                        sumo_sim.ResetDetectedVehicle() # 주기 5,10 인 경우 누적된 상태가 들어가면 안되므로 주기마다 시뮬레이션 결과를 초기화하는 함수
                    _tempPattern:Pattern = _newTOD.GetPatternFromIndex(_action_Index)
                    _dbID = _tempPattern.GetPatternID()
                
                    _linkQueueCountDic= _simulResult['linkvehstopqueuecounts'] # 링크별 대기열 (차량 대수)
                    _linkInCountDic = _simulResult['linkincounts'] # 링크별 유입량
                    _linkOutCountDic = _simulResult['linkoutcounts'] # 링크별 유출량
                    _linkPassTimeDic = _simulResult['linkmeanpasstimes'] # 링크별 평균 통과시간
                    _linkDensityDic = _simulResult['linkmaxdensitys'] # 링크별 최대 밀도
                    _linkSpeedDic = _simulResult['linkmeanspeeds'] # 링크별 평균 속도
                    _cycleSumPassTimeDic = _simulResult['linksumpasstimes'] # 링크별 총 통과시간
                    
                    _timeStr = str(datetime.timedelta(seconds=_currentTime))
                    # State 생성
                    def GetState_Reward(_beforeReward:dict):
                        # 0~ 23 시  / 0 : 월 ~ 6 : 일요일 / 0 : 국경일X , 1 : 국경일o
                        _nextStates.extend([_currentTime//3600,_day.weekday(),_isNationalDay])
                        _inCountSum = 0
                        _passCountSum = 0
                        _cycleSumPassTime = 0
                        _avgSpeed = 0
                        _nowReward = 0
                        
                        for _key in _linkInCountDic.keys():
                            _queueCount = _linkQueueCountDic[_key]
                            _inCount = _linkInCountDic[_key]
                            _finalInCount = _inCount + _queueCount # 총 유입량 = 유입량 + 대기열 길이
                            _inCountSum += _inCount
                            _outCount = _linkOutCountDic[_key]
                            _passCountSum += _outCount
                            _passTime = _linkPassTimeDic[_key]
                            _speed = _linkSpeedDic[_key]
                            _avgSpeed += _speed
                            _cycleSumPassTime += _cycleSumPassTimeDic[_key]
                            if _finalInCount == 0 :
                                _cong = 0
                            else:
                                _cong = _outCount/_finalInCount
                            
                            _nowReward += _beforeReward[_key] - _linkDensityDic[_key] # 전 밀도 - 다음 밀도 
                            
                            # 링크별 : 실제 유입량, 대기열 수, 유출량, 밀도, 통과시간 합
                            _statesRawDataList.extend([_inCount,_queueCount,_outCount,_linkDensityDic[_key],_cycleSumPassTimeDic[_key]])    
                            # 링크별 평균속도, 통과시간, 혼잡강도
                            _nextStates.extend([_speed,_passTime,round(_cong,3)])
                        _beforeReward = _linkDensityDic

                        return _inCountSum,_passCountSum, round(_cycleSumPassTime,3), round(_avgSpeed/len(_linkSpeedDic),3), round(_nowReward,3) , _beforeReward
                        
                    _inCountSum,_passCountSum,_cycleSumPassTime,_avgSpeed, _reward, _beforeReward = GetState_Reward(_beforeReward)
                    
                    _nextStates = np.array(_nextStates)
                    # 메모리에 기록
                    _done_mask = 1.0 if _isDone else 0.0
                    dqnManager.PushMemory(_state, _action_Index, _reward, _nextStates, _done_mask)

                    _state = _nextStates # 다음상태가 시뮬레이션으로 다시 적용
                    _score += _reward # _score : 하루 누적 reward 값
                    _accumulatePassTime += _cycleSumPassTime  # __accumulatePassTime : 하루 누적 통과시간 
                    
                    _intersectionState = ""
                    for i in range(0,len(_nextStates)):
                        if(i == len(_nextStates)-1):
                            _intersectionState += str(_nextStates[i])
                            break
                        _intersectionState += str(_nextStates[i])+","
                        
                    _intersectionStateRawStr = ""
                    for i in range(0,len(_statesRawDataList)):
                        if(i == len(_statesRawDataList)-1):
                            _intersectionStateRawStr += str(_statesRawDataList[i])
                            break
                        _intersectionStateRawStr += str(_statesRawDataList[i])+","
                    
                    if _passCountSum == 0 : # 0으로 나눌 수 없으므로 예외처리 
                        _vehicleAvgPassTime = 0
                    else :     
                        _vehicleAvgPassTime = _cycleSumPassTime/_passCountSum
                    
                    if _modelMode == ModelMode.TRAIN:
                        dqnManager.TrainModel() # 주기한번 돌때마다 모델 학습
                        _tdMgr.InsertLearnData(_epiID,_timeStr,round(_reward,3),
                                               _cycleSumPassTime,_passCountSum,_action_Index,
                                               _intersectionState,_intersectionStateRawStr,
                                               _intersectionID,_avgSpeed,_vehicleAvgPassTime,_inCountSum)
                    else:
                        _startDate = datetime.datetime.today().strftime('%Y-%m-%d')
                        _tdMgr.InsertPredictLearnData(_predictID,_startDate,_timeStr,round(_reward,3),
                                                      _cycleSumPassTime,_passCountSum,_action_Index,
                                                      _intersectionState,_intersectionStateRawStr,
                                                      _intersectionID,_avgSpeed,_vehicleAvgPassTime,_inCountSum)

                    print("Episode :", epi, "ActionIndex :", _action_Index, "Percent :", _percent)
                    print(f"Running {_target}/{_currentTime}\n")
                except Exception as _ex:    
                    _episodeState = 'FAIL'
                    print("에러발생", traceback.format_exc())
                    sumo_sim.Stop()
                    raise _ex
                finally:
                    if _modelMode == ModelMode.TRAIN:
                        _tdMgr.UpdateEpisode(_epiID,round(_accumulatePassTime,3),round(_score,3),dqnManager.MemorySize(),_episodeState)
                    else:
                        _tdMgr.UpdatePredictInfo(_predictID,round(_accumulatePassTime,3),round(_score,3))
                    _tdMgr.Commit()
            sumo_sim.Stop()
            print("epi : {}, score : {:.1f}, memorySize : {}, eps : {:.3f}%, accumlatePassTime : {}".format(
                    epi, _score, dqnManager.MemorySize(), _epsilon*100, round(_accumulatePassTime,3)))
            
            if(_modelMode == ModelMode.TRAIN):
                if(best_score < _score): # 학습 데이터에서 최대 하루 누적 reward 비교
                    best_score = _score
                    _tdMgr.PeriodUpdateHistory(_historyID,round(best_score,3))
                
            _accumulatePassTime = 0
            _score = 0
        
            #에피소드 별 모델 검증 후 best 모델 저장, reward로 할지 통과시간으로 할지
            if(_modelMode == ModelMode.TRAIN):
                if epi % 50 == 0 and epi != 0: # epi 50 배수마다 모델 저장
                    dqnManager.Savemodel(PredictMode.SELECT,epi)
                if(dqnManager.MemorySize() >= Config.train_size): 
                    _valVehicleAvgPassTime , _avgReward = RunRLValidation(dqnManager,_newTOD,_valData)
                    if(_valPassTime > _valVehicleAvgPassTime): # 현재는 최고 성능의 모델 추출 방법을 하루 차량당 평균 통과시간을 이용 / _avgReward로 할때 로직 구현 필요
                        _valPassTime = _valVehicleAvgPassTime
                        dqnManager.Savemodel(PredictMode.BEST)
                        _tdMgr.UpdateHistoryValData(_historyID,
                                                    round(_valVehicleAvgPassTime,3),
                                                    round(_avgReward,3),
                                                    epi)
                
    except KeyboardInterrupt as _ex:
        print("KeyBoardInterrrupt 에러발생", _ex)
        _historyState = 'FAIL'
        _errorMsg = traceback.format_exc()
    except Exception as ex:
        print("에러발생", traceback.format_exc())
        _historyState = 'FAIL'
        _errorMsg = traceback.format_exc()
    finally:
        if not dqnManager is None:
            if(_modelMode == ModelMode.TRAIN): # 모델 학습 시 최종 epi의 모델 저장
                dqnManager.Savemodel(PredictMode.FINAL)
                dqnManager.tensorWriter.close()
                
        if _tdMgr != None:
            _tdMgr.Commit()
            if _modelMode == ModelMode.TRAIN:
                _learningEndTimeStr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                _tdMgr.OneUpdateHistory(_historyID,_learningEndTimeStr,_historyState,_errorMsg)

        
            #dqnManager.trainProcessVisualization()

def RunNormal(_todMode = TODMode.Manual, _simulMode=sMode.Fixed, _weekType = PlanType.WeekDay):
    from Util import CSVManager as csvMgr
    _score = 0
    _accumulatePassTime = 0
    try:
        # TOD DB 생성
        _tlsMgr = TLSManager()
        _newTOD: TOD = _tlsMgr.GetTOD(Config.NODE_ID, Config.NodeName)
        _actionCount = _newTOD.GetPatternCount()
        
        _tdMgr = TraffciDataManager(Config._data_DBName, Config._ip, Config._port, Config._id, Config._pw)
        
        _data = pd.read_csv(Config._vdsFilePath)
        _data[['DAY', 'TIME']] = _data['TOT_DT'].str.split(' ',n=1, expand=True)
        _vdsDayGroup = _data.groupby(['DAY'])

        _dataMgr = DataManager()
        _dataMgr.Initialize()
        _intersectionID = _dataMgr.GetIntersectionID(Config.NODE_ID)
        
        sumo_sim = SumoSimulator( _tod=_newTOD, _todMode=_todMode, _sMode=_simulMode, _patternIndex=1, _rewardType=RewardType.DensityAndPassTime)
        _groupKeyList = list(_vdsDayGroup.groups.keys())
        _gruopIndex = 0

        _state = [0 for i in(range(Config.modelInputSize))]
        
        _modelID = _tdMgr.SelectModel(Config._finalModelName)
        if _modelID is None:
            _modelID = _tdMgr.InsertModel(Config._finalModelName,len(_state), _actionCount, Config.buffer_limit,
                                          Config.learning_rate,Config.batch_size,
                                          Config.updatePeriod,Config.gamma,Config.train_size,
                                        Config._eps_start,Config._eps_end,Config._episodes)
        else :
            _tdMgr.UpdateModel(_modelID,len(_state), _actionCount, Config.buffer_limit,
                                          Config.learning_rate,Config.batch_size,
                                          Config.updatePeriod,Config.gamma,Config.train_size,
                                        Config._eps_start,Config._eps_end,Config._episodes)
            
        _fixedID = _tdMgr.InsertFixedInfo(_modelID, 'FIXED', _groupKeyList[0])
        
        _beforeReward = {}
        _beforeReward['충청대학교 방향'] = 0
        _beforeReward['오송역 방향'] = 0
        _beforeReward['청주역 방향'] = 0
        _beforeReward['청주IC 방향'] = 0
        #for epi in range(Config._episodes):
        for epi in range(1):
            _timeGroup = _vdsDayGroup.get_group(_groupKeyList[_gruopIndex]).groupby(['TIME'])
            if not SumoUtil.CreateTripFile(_timeGroup):
                print('Trip 파일 생성 실패')

            _target = len(_timeGroup) * 900
            sumo_sim.Start(_groupKeyList[_gruopIndex], _target)

            dateTimeStr = _groupKeyList[_gruopIndex]
            
            _gruopIndex += 1
            if _gruopIndex >= len(_groupKeyList):
                _gruopIndex = 0

            _datetime = datetime.datetime.strptime(dateTimeStr, '%Y-%m-%d')
            _day = _datetime.date()
            nationalDayList = Env_Sumo.GetNationalDay()
            if _day in nationalDayList:
                _isNationalDay = 1
            else :
                _isNationalDay = 0
                
            for _currentTime in sumo_sim:
                _nextStates = []
                _statesRawDataList = []
                
                for i in range(Config.LOOP_COUNT):
                    _simulResult, _isDone = sumo_sim.RunCycleSimulation()
                    sumo_sim.ResetDetectedVehicle()
                _linkQueueCountDic= _simulResult['linkvehstopqueuecounts']
                _linkInCountDic = _simulResult['linkincounts']
                _linkOutCountDic = _simulResult['linkoutcounts']
                _linkPassTimeDic = _simulResult['linkmeanpasstimes']
                _linkDensityDic = _simulResult['linkmaxdensitys']
                _linkSpeedDic = _simulResult['linkmeanspeeds']
                _cycleSumPassTimeDic = _simulResult['linksumpasstimes']
                
                # State 생성
                def GetState_Reward(_beforeReward:dict):
                    # 0 : 월 ~ 6 : 일요일 / 0 : 국경일X , 1 : 국경일o
                    _nextStates.extend([_currentTime//3600,_day.weekday(),_isNationalDay])
                    _inCountSum = 0
                    _passCountSum = 0
                    _cycleSumPassTime = 0
                    _avgSpeed = 0
                    _nowReward = 0
                    
                    for _key in _linkInCountDic.keys():
                        _queueCount = _linkQueueCountDic[_key]
                        _inCount = _linkInCountDic[_key]
                        _finalInCount = _inCount + _queueCount 
                        _inCountSum += _inCount
                        _outCount = _linkOutCountDic[_key]
                        _passCountSum += _outCount
                        _passTime = _linkPassTimeDic[_key]
                        _speed = _linkSpeedDic[_key]
                        _avgSpeed += _speed
                        _cycleSumPassTime += _cycleSumPassTimeDic[_key]
                        if _finalInCount == 0 :
                            _cong = 0
                        else:
                            _cong = _outCount/_finalInCount
                        
                        _nowReward += _beforeReward[_key] - _linkDensityDic[_key]
                        
                        # 링크별 : 실제 유입량, 대기열 수, 유출량, 밀도, 평균통과시간, 통과시간 합
                        _statesRawDataList.extend([_inCount,_queueCount,_outCount,_linkDensityDic[_key],_cycleSumPassTimeDic[_key]])    
                        
                        _nextStates.extend([_speed,_passTime,round(_cong,3)])
                    _beforeReward = _linkDensityDic

                    return _inCountSum,_passCountSum, round(_cycleSumPassTime,3), round(_avgSpeed/len(_linkSpeedDic),3), round(_nowReward,3) , _beforeReward
                        
                _inCountSum,_passCountSum,_cycleSumPassTime,_avgSpeed, _reward, _beforeReward = GetState_Reward(_beforeReward)

                _score += _reward
                _accumulatePassTime += _cycleSumPassTime
                print("Episode :", epi, f"Running {_target}/{_currentTime}")
                print()
       
                _intersectionState = ""
                for i in range(0,len(_nextStates)):
                    if(i == len(_nextStates)-1):
                        _intersectionState += str(_nextStates[i])
                        break
                    _intersectionState += str(_nextStates[i])+"," 
                        
                _intersectionStateRawStr = ""
                for i in range(0,len(_statesRawDataList)):
                    if(i == len(_statesRawDataList)-1):
                        _intersectionStateRawStr += str(_statesRawDataList[i])
                        break
                    _intersectionStateRawStr += str(_statesRawDataList[i])+","
                
                if _passCountSum == 0 :
                    _vehicleAvgPassTime = 0
                else :     
                    _vehicleAvgPassTime = _cycleSumPassTime/_passCountSum
                
                _tmpPattern = sumo_sim.GetPattern()
                _action_Index = _newTOD.GetPatternIndex(_tmpPattern)
                _startDate = datetime.datetime.today().strftime('%Y-%m-%d')
                _timeStr = str(datetime.timedelta(seconds=_currentTime))
                _tdMgr.InsertFixedLearnData(_fixedID,
                                              _startDate,
                                              _timeStr,
                                              round(_reward,3),
                                              _cycleSumPassTime,
                                              _passCountSum,
                                              _action_Index,
                                              _intersectionState,
                                              _intersectionStateRawStr,
                                              _intersectionID,
                                              _avgSpeed,
                                              _vehicleAvgPassTime,
                                              _inCountSum)
                
            sumo_sim.Stop()
            print(f"epi : {epi}, score : {_score:.1f}, accumlatePassTime : {_accumulatePassTime}")

        _tdMgr.UpdateFixedInfo(_fixedID,_accumulatePassTime,_score)
        _accumulatePassTime = 0
        _score = 0
        
    except KeyboardInterrupt as _ex:
        print("KeyBoardInterrrupt 에러발생", _ex)
    except Exception as ex:
        print("에러발생", traceback.format_exc())
    finally:
        print()


def RunRLValidation(_dqnManager:DQN_Manager, _newTOD:TOD,_valData:pd.DataFrame,_todMode = TODMode.Manual, _simulMode=sMode.Fixed):
    _score = 0
    _accumulatePassTime = 0
    _accumulateOutCount = 0
    _start_epi = 0
    _vehicleAvgPassTimeList = []
    _rewardList = []
    
    _actionCount = _newTOD.GetPatternCount()
    
    sumo_sim = SumoSimulator(_todMode=_todMode, _sMode=_simulMode, _tod=_newTOD, _rewardType=RewardType.DensityAndPassTime)
    
    _state = [0 for i in(range(Config.modelInputSize))]
    _state = np.array(_state)
        
    _valData[['DAY', 'TIME']] = _valData['TOT_DT'].str.split(' ',n=1, expand=True)
    _vdsDayGroup = _valData.groupby(['DAY'])
    
    _groupKeyList = list(_vdsDayGroup.groups.keys())
    _gruopIndex = 0
    
    for epi in range(_start_epi, len(_groupKeyList)):
        _beforeReward = {}
        _beforeReward['충청대학교 방향'] = 0
        _beforeReward['오송역 방향'] = 0
        _beforeReward['청주역 방향'] = 0
        _beforeReward['청주IC 방향'] = 0
        dateTimeStr = _groupKeyList[_gruopIndex]
        _timeGroup = _vdsDayGroup.get_group(_groupKeyList[_gruopIndex]).groupby(['TIME'])
        
        if not SumoUtil.CreateTripFile(_timeGroup):
            print('Trip 파일 생성 실패')

        _target = len(_timeGroup) * 900
        sumo_sim.Start(_groupKeyList[_gruopIndex], _target)

        _gruopIndex += 1
        if _gruopIndex >= len(_groupKeyList):
            _gruopIndex = 0

        _datetime = datetime.datetime.strptime(dateTimeStr, '%Y-%m-%d')
        _day = _datetime.date()
        nationalDayList = Env_Sumo.GetNationalDay()
        if _day in nationalDayList:
            _isNationalDay = 1
        else :
            _isNationalDay = 0
        
        for _currentTime in sumo_sim:
            _nextStates = []
            _statesRawDataList = []

            _action_Index, _percent = _dqnManager.Select_action(_state, 0, _actionCount-1,ModelMode.INFERENCE)
            
            for i in range(Config.LOOP_COUNT):
                _simulResult, _isDone = sumo_sim.RunCycleSimulation(_action_Index)
                sumo_sim.ResetDetectedVehicle()
            _linkQueueCountDic= _simulResult['linkvehstopqueuecounts']
            _linkInCountDic = _simulResult['linkincounts']
            _linkOutCountDic = _simulResult['linkoutcounts']
            _linkPassTimeDic = _simulResult['linkmeanpasstimes']
            _linkDensityDic = _simulResult['linkmaxdensitys']
            _linkSpeedDic = _simulResult['linkmeanspeeds']
            _cycleSumPassTimeDic = _simulResult['linksumpasstimes']
                    
            def GetState_Reward(_beforeReward:dict):
                # 0 : 월 ~ 6 : 일요일 / 0 : 국경일X , 1 : 국경일o
                _nextStates.extend([_currentTime//3600,_day.weekday(),_isNationalDay])
                _inCountSum = 0
                _passCountSum = 0
                _cycleSumPassTime = 0
                _avgSpeed = 0
                _nowReward = 0
                
                for _key in _linkInCountDic.keys():
                    _queueCount = _linkQueueCountDic[_key]
                    _inCount = _linkInCountDic[_key]
                    _finalInCount = _inCount + _queueCount 
                    _inCountSum += _inCount
                    _outCount = _linkOutCountDic[_key]
                    _passCountSum += _outCount
                    _passTime = _linkPassTimeDic[_key]
                    _speed = _linkSpeedDic[_key]
                    _avgSpeed += _speed
                    _cycleSumPassTime += _cycleSumPassTimeDic[_key]
                    if _finalInCount == 0 :
                        _cong = 0
                    else:
                        _cong = _outCount/_finalInCount
                    
                    _nowReward += _beforeReward[_key] - _linkDensityDic[_key]
                    
                    _nextStates.extend([_speed,_passTime,round(_cong,3)])
                
                _beforeReward = _linkDensityDic
                return _inCountSum,_passCountSum, round(_cycleSumPassTime,3), round(_avgSpeed/len(_linkSpeedDic),3), round(_nowReward,3) , _beforeReward
                
            _inCountSum, _passCountSum, _cycleSumPassTime, _avgSpeed, _reward, _beforeReward = GetState_Reward(_beforeReward)

            _state = np.array(_nextStates)
            _score += _reward
            _accumulatePassTime += _cycleSumPassTime 
            _accumulateOutCount += _passCountSum

            print("Validation ----------- Episode :", epi, "ActionIndex :", _action_Index, "Percent :", _percent)
            print(f"Validation ----------- Running {_target}/{_currentTime}\n")
            
        sumo_sim.Stop()
        
        print("Validation ----------- epi : {}, score : {:.1f}, memorySize : {}, accumlatePassTime : {}".format(
                epi, _score, _dqnManager.MemorySize(), round(_accumulatePassTime,3)))
        
        _vehicleAvgPassTime = _accumulatePassTime/_accumulateOutCount
        
        _vehicleAvgPassTimeList.append(_vehicleAvgPassTime)
        _rewardList.append(_score)
        
        _accumulatePassTime = 0
        _accumulateOutCount = 0
        _score = 0
    # return 차량 당 평균 통과시간, 하루 누적 평균 Reward
    return np.mean(_vehicleAvgPassTimeList), np.mean(_rewardList)

if __name__ == "__main__":
    RunRL(_modelMode=ModelMode.TRAIN,_todMode=TODMode.Manual, _simulMode=sMode.Random, _predictMode = PredictMode.BEST)
    #RunRL(_modelMode=ModelMode.INFERENCE,_predictMode=PredictMode.BEST, _todMode=TODMode.Manual, _simulMode=sMode.Fixed, _planType=PlanType.WeekDay)
    #RunRL(_modelMode=ModelMode.INFERENCE,_predictMode=PredictMode.FINAL, _todMode=TODMode.Manual, _simulMode=sMode.Fixed, _planType=PlanType.WeekDay)
    #RunNormal(_todMode=TODMode.Auto, _simulMode=sMode.Fixed)