import csv
import os
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import datetime

'''
그래프 그릴때 사용되는 클래스
 - DF의 속성명
'''
class TrafficFactor(Enum):
    PassVehicleCount = 'PhasePassCount'
    PassTime = 'PhasePassTime'
    AvgDensity = 'LinkMeanDensitysAvgVal'
    MaxDensity = 'LinkMaxDensitysAvgVal'


class CSVManager:
    def __init__(self):
        self.__file = None

    def Open(self, _dir:str, _fileName = 'model_history.csv',
             _headers:list = ['Episode','ExplorationRate','Reward','AccumlatePassTime'], _oType:str = 'a'):
        self._filePath = os.path.join(_dir, _fileName)
        _isFile = False
        if os.path.isfile(self._filePath):
            _isFile = True
        self.__file = open(self._filePath, _oType, encoding='utf-8', newline='')
        self._wr = csv.writer(self.__file)
        if not _isFile or _oType == 'w':
            self._wr.writerow(_headers)

    def Close(self):
        self.__file.close()

    def Write(self, _dataList):
        self._wr.writerow(_dataList)
        self.__file.flush()


def Plot(_file1, _plot):
    _historyDf1 = None

    try:
        _historyDf1 = pd.read_csv(_file1, encoding='utf-8', sep = ',')
    except Exception as _ex:
        print(_ex)
        return

    _timeList = _historyDf1['Time'].tolist()
    _PassTimeList = _historyDf1['PhasePassTime'].tolist()

    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("PhasePassTime")
    line1 = ax.plot(_timeList, _PassTimeList, color = 'green', label='Normal')

    lines = line1
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper right')

    fig = plt.gcf()
    fig.set_size_inches(40,20)
    plt.savefig(_plot, bbox_inches='tight', pad_inches=0)

# 그래프 하나에 두개
def Plot2(_inputModel:str,_inputFixed:str,_outGraph:str):
    _modelDf = None
    _fixDf = None
    try:

        _modelDf = pd.read_csv(_inputModel, encoding='utf-8', sep = ',')
        _fixDf = pd.read_csv(_inputFixed, encoding='utf-8', sep = ',')
    except Exception as _ex:
        print(_ex)
        return

    _timeVal = 60

    _modelDf['Tag'] = [int(e/_timeVal)*_timeVal for e in _modelDf.Time]
    _groups = _modelDf.groupby(['Tag'])
    _modelPassTimeList = _groups['PhasePassTime'].mean(numeric_only = True).tolist()
    _modelTimeList = [_a[0] for _a in _groups.Tag.unique()]
    
    _fixDf['Tag'] = [int(e/_timeVal)*_timeVal for e in _fixDf.Time]
    _groups = _fixDf.groupby(['Tag'])
    _fixPassTimeList = _groups['PhasePassTime'].mean(numeric_only = True).tolist()
    _fixTimeList = [_a[0] for _a in _groups.Tag.unique()]

    plt.figure(figsize=(40,20))

    plt.plot(_modelTimeList, _modelPassTimeList, color = 'green', label='Model')
    plt.plot(_fixTimeList, _fixPassTimeList, color = 'lightcoral', label='Fixed')
    plt.xlabel("Time")
    plt.ylabel("PhasePassTime")
    plt.legend()

    plt.draw()
    plt.savefig(fname=_outGraph, bbox_inches='tight', pad_inches=0)

def TrafficRawFactorPlot(_inputModelPath:str,_outGraphPath:str,_trafficFactor:str):
    _modelDf = pd.read_csv(_inputModelPath,encoding='utf-8',sep = ',')
    _modelDf['StartTime'] = _modelDf['StartTime'].str[7:]
    
    def strToTimedelta(_strTimes):
        _strTimeList = _strTimes.split(':')
        _time = datetime.timedelta(hours=int(_strTimeList[0]),minutes=int(_strTimeList[1]),seconds=int(_strTimeList[2]))    
        return _time
    
    _modelDf['StartTime'] = _modelDf['StartTime'].apply(strToTimedelta)
    
    _modelDf_Before = _modelDf[(_modelDf['StartTime'] >= datetime.timedelta(hours=7)) & (_modelDf['StartTime'] <= datetime.timedelta(hours=10))]
    _modelDf_After = _modelDf[(_modelDf['StartTime'] >= datetime.timedelta(hours=17)) & (_modelDf['StartTime'] <= datetime.timedelta(hours=20))]
    
    _modelDf_Before = _modelDf_Before.astype({'StartTime':'str'})
    _modelDf_Before['StartTime'] = _modelDf_Before['StartTime'].str[7:]
    
    _modelDf_After = _modelDf_After.astype({'StartTime':'str'})
    _modelDf_After['StartTime'] = _modelDf_After['StartTime'].str[7:]
    
    colNameStr = ['Link1_','Link2_','Link3_','Link4_']
    colFullNameStr = []
    for factor in colNameStr:
        colFullNameStr.append(factor+_trafficFactor)      
         
    color =['green','red','blue','black']
    
    plt.figure(figsize=(40,20))

    plt.subplot(2,1,1)
    for index, factor in enumerate(colFullNameStr):
        plt.plot(_modelDf_Before['StartTime'], _modelDf_Before[factor],marker='o', color = color[index], label= factor)
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.ylabel(factor)
    plt.title('Morning PeakTime')
    plt.legend()

    plt.subplot(2,1,2)
    for index, factor in enumerate(colFullNameStr):
        plt.plot(_modelDf_After['StartTime'], _modelDf_After[factor],marker='o', color = color[index], label= factor)
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.ylabel(factor)
    plt.title('Afternoon PeakTime')
    plt.legend()

    plt.draw()
    plt.savefig(fname=_outGraphPath+'_RawData_'+_trafficFactor +'.png', bbox_inches='tight', pad_inches=0)

def TrafficFactorPlot(_inputModel:str,_inputFixed:str,_outGraph:str,_fileNickName:str):
    _modelDf = None
    _fixDf = None
    try:
        _modelDf = pd.read_csv(_inputModel, encoding='utf-8', sep = ',')
        _fixDf = pd.read_csv(_inputFixed, encoding='utf-8', sep = ',')
    except Exception as _ex:
        print(_ex)
        return
   
    #_timeStr = str(datetime.timedelta(seconds=_currentTime))
    # 오전 7 ~ 10시 , 오후 5 ~ 8시
    _modelDf_Before = _modelDf[((_modelDf['Time'] >= 25200) & (_modelDf['Time'] <= 36000))]
    _fixDf_Before = _fixDf[((_fixDf['Time'] >= 25200) & (_fixDf['Time'] <= 36000))]
    
    _modelDf_After = _modelDf[((_modelDf['Time'] >= 61200) & (_modelDf['Time'] <= 72000))]
    _fixDf_After = _fixDf[((_fixDf['Time'] >= 61200) & (_fixDf['Time'] <= 72000))]
    
    _trafficFactors = ['PhasePassCount','PhasePassTime']#,'LinkMaxDensitysAvgVal','LinkMeanDensitysAvgVal']
    
    for _trafficFactor in _trafficFactors:
        _morningModelDfTrafficVal = _modelDf_Before[_trafficFactor].sum()
        _morningfixDfTrafficVal = _fixDf_Before[_trafficFactor].sum()
        
        _afterModelDfTrafficVal = _modelDf_After[_trafficFactor].sum()
        _afterFixDfTrafficVal = _fixDf_After[_trafficFactor].sum()
        
        _morningStateVal = (_morningfixDfTrafficVal - _morningModelDfTrafficVal) / _morningfixDfTrafficVal * 100
        _afterStateVal = (_afterFixDfTrafficVal - _afterModelDfTrafficVal) / _afterFixDfTrafficVal * 100
        
        if(_trafficFactor == 'PhasePassCount'):
            print("오전 모델 통과차량 수 총합 : ",_morningModelDfTrafficVal)
            print("오전 고정주기 통과차량 수 총합 : ",_morningfixDfTrafficVal)
            print("오전 통과차량 수 증가율 : ",_morningStateVal)
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            print("오후 모델 통과차량 수 총합 : ",_afterModelDfTrafficVal)
            print("오후 고정주기 통과차량 수 총합 : ",_afterFixDfTrafficVal)
            print("오후 통과차량 수 증가율 : ",_afterStateVal)
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            
        elif(_trafficFactor == 'PhasePassTime'):
            print("오전 모델 차량 통과시간 총합",_morningModelDfTrafficVal)
            print("오전 고정주기 차량 통과시간 총합",_morningfixDfTrafficVal)   
            print("오전 통과시간 감소율 : ",_morningStateVal)
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            print("오후 모델 차량 통과시간 총합",_afterModelDfTrafficVal)
            print("오후 고정주기 차량 통과시간 총합",_afterFixDfTrafficVal)   
            print("오후 통과시간 감소율 : ",_afterStateVal)
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            
        elif(_trafficFactor == 'LinkMeanDensitysAvgVal'):
            print("오전 모델 차량 평균밀도 총합", _morningModelDfTrafficVal)     
            print("오전 고정주기 차량 평균밀도 총합", _morningfixDfTrafficVal)     
            print("오전 차량 평균밀도 감소율 : ",_morningStateVal)
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            print("오후 모델 차량 평균밀도 총합", _afterModelDfTrafficVal)     
            print("오후 고정주기 차량 평균밀도 총합", _afterFixDfTrafficVal)     
            print("오후 차량 평균밀도 감소율 : ",_afterStateVal)
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            
        elif(_trafficFactor == 'LinkMaxDensitysAvgVal'):
            print("오전 모델 차량 최대밀도 총합 : ", _morningModelDfTrafficVal)
            print("오전 고정주기 차량 최대밀도 총합 : ", _morningfixDfTrafficVal)
            print("오전 차량 최대밀도 감소율 : ",_morningStateVal)    
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            print("오후 모델 차량 최대밀도 총합 : ", _afterModelDfTrafficVal)
            print("오후 고정주기 차량 최대밀도 총합 : ", _afterFixDfTrafficVal)
            print("오후 차량 최대밀도 감소율 : ",_afterStateVal) 
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
            
        plt.figure(figsize=(40,20))

        plt.subplot(2,1,1)
        plt.plot(_modelDf_Before['Time'], _modelDf_Before[_trafficFactor],marker='o', color = 'green', label='Model')
        plt.plot(_fixDf_Before['Time'], _fixDf_Before[_trafficFactor],marker='o' , color = 'lightcoral', label='Fixed')
        plt.xlabel("Time")
        plt.ylabel(_trafficFactor)
        plt.title('Morning PeakTime')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(_modelDf_After['Time'], _modelDf_After[_trafficFactor],marker='o', color = 'green', label='Model')
        plt.plot(_fixDf_After['Time'], _fixDf_After[_trafficFactor],marker='o', color = 'lightcoral', label='Fixed')
        plt.xlabel("Time")
        plt.ylabel(_trafficFactor)
        plt.title('Afternoon PeakTime')
        plt.legend()

        #plt.tight_layout()
        plt.draw()
        plt.savefig(fname=_outGraph+'_'+_trafficFactor + _fileNickName +'.png', bbox_inches='tight', pad_inches=0)
    
def TrafficOneFactorPlot(_inputModel:str,_inputFixed:str,_outGraph:str,_trafficFactor:TrafficFactor):
    _modelDf = None
    _fixDf = None
    try:
        _modelDf = pd.read_csv(_inputModel, encoding='utf-8', sep = ',')
        _fixDf = pd.read_csv(_inputFixed, encoding='utf-8', sep = ',')
    except Exception as _ex:
        print(_ex)
        return
   
    # 오전 10시 ~ 오후 5시
    _modelDf_Before = _modelDf[((_modelDf['Time'] >= 36000) & (_modelDf['Time'] <= 72000))]
    _fixDf_Before = _fixDf[((_fixDf['Time'] >= 36000) & (_fixDf['Time'] <= 72000))]
    
    _modelDfTrafficVal = _modelDf_Before[_trafficFactor.value].sum()
    _fixDfTrafficVal = _fixDf_Before[_trafficFactor.value].sum()
    _StateVal = 1-(_modelDfTrafficVal/_fixDfTrafficVal * 100)
    if(_trafficFactor == TrafficFactor.PassVehicleCount):
        print("모델 통과차량 수 총합 : ",_modelDfTrafficVal)
        print("고정주기 통과차량 수 총합 : ",_fixDfTrafficVal)
        #print("통과차량 수 증가율 : ",_StateVal)
    elif(_trafficFactor == TrafficFactor.PassTime):
        print("모델 차량 통과시간 총합",_modelDfTrafficVal)
        print("고정주기 차량 통과시간 총합",_fixDfTrafficVal)   
        #print("통과시간 감소율 : ",_StateVal)
    elif(_trafficFactor == TrafficFactor.AvgDensity):
        print("모델 차량 평균밀도 총합", _modelDfTrafficVal)     
        print("고정주기 차량 평균밀도 총합", _fixDfTrafficVal)     
        #print("차량 평균밀도 감소율 : ",_StateVal)
    elif(_trafficFactor == TrafficFactor.MaxDensity):
        print("모델 차량 최대밀도 총합 : ", _modelDfTrafficVal)
        print("고정주기 차량 최대밀도 총합 : ", _fixDfTrafficVal)
        #print("차량 최대밀도 감소율 : ",_StateVal)    

    plt.figure(figsize=(40,20))

    plt.subplot(2,1,1)
    plt.plot(_modelDf_Before['Time'], _modelDf_Before[_trafficFactor.value],marker='o', color = 'green', label='Model')
    plt.plot(_fixDf_Before['Time'], _fixDf_Before[_trafficFactor.value],marker='o' , color = 'lightcoral', label='Fixed')
    plt.xlabel("Time")
    plt.ylabel(_trafficFactor.value)
    plt.title('10:00 ~ 17:00')
    plt.legend()

    plt.draw()
    plt.savefig(fname=_outGraph+'_'+_trafficFactor.value+'_비첨두시간.png', bbox_inches='tight', pad_inches=0)   
     
def GetAccmulatePassTime(_fileDir):
    _df = pd.read_csv(_fileDir,encoding='utf-8', sep = ',')
    _accmulatePassTimeSum = _df['PhasePassTime'].sum()
    return _accmulatePassTimeSum

def GetAccmulateVehicleCount(_fileDir):
    _df = pd.read_csv(_fileDir,encoding='utf-8', sep = ',')
    _accmulatePassVehicleSum = _df['PhasePassCount'].sum()
    return _accmulatePassVehicleSum

def CheckData(_filepath):
    _df = pd.read_csv(_filepath)
    #_df = _df[['ACSR_ID']=='']
    _df = CheckData(Config._basePath + '/VDSData/spin_20220122.csv')
    _baseTime = datetime.datetime(1900,1,1,0,0,0)
    _simdf = pd.read_excel(Config._basePath+'/VDSData/20220122_simulation.xlsx')
    _simdf['StartTime'] = _simdf['StartTime'].str[7:]
    _simdf['StartTime'] = pd.to_datetime(_simdf['StartTime'],format= '%H:%M:%S')
    _simdf['StartTime'] = _simdf['StartTime'] - _baseTime
    
    #_simdf['StartTime'] = _simdf['StartTime'].dt.time
    
    #timeStr = '00:00:00'
    #print((_simdf['StartTime'][0]) + datetime.datetime.timedelta(minutes=15))
    
    print(_simdf.head())
    print(_simdf.info())
        
    #print(_simdf['StartTime'])
    #print("총 데이터 개수 :", len(_simdf))
    #_df = _df[(_df['ACSR_ID']=='오송역 방향') & (_df['AVG_SPD'] < 100)]
    #print(_df.head())
    #print(_df.head())
    #print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
    #_df.loc[_df['AVG_SPD'] >= 100 ,'AVG_SPD'] =  _df[['AVG_SPD']]%100
    #print(_df.head())
    return _df
    
if __name__ == "__main__":
    import Config
    
    _fileName1 = 'Predict_Simulation_History_model_DQN_Reward_Density_final_2배_75_20220122.csv'
    _fixName = 'Fixed_Simulation_History_75_20220122.csv'
    TrafficFactorPlot(Config._outPredictPath+_fileName1, Config._outPredictPath+_fixName, Config.TrafficFactorPath,'*2')
    #TrafficOneFactorPlot(Config._outPredictPath+_fileName, Config._outPredictPath+_fixName, Config.TrafficFactorPath,TrafficFactor.AvgDensity)
    
    print("새로운모델 통과시간: ",GetAccmulatePassTime(Config._outPredictPath+_fileName1))
    print("기존TOD모델 하루 통과시간: ",GetAccmulatePassTime(Config._outPredictPath+_fixName))
    
    
    print("새로운모델 : ",GetAccmulateVehicleCount(Config._outPredictPath+_fileName1))
    print("기존TOD모델 : ",GetAccmulateVehicleCount(Config._outPredictPath+_fixName))
    
    #[MeanSpeed,MeanPassTime,SumPassTime,Cong,InCount,QueueCount,OutCount,Density]
    #TrafficRawFactorPlot(Config._basePath+'/DataCheck.csv',Config._basePath+'/RawDataGraph/','Cong')
    