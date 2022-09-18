import datetime as dt
from mariaDBSql import MariaDBSql as mdb
import pandas as pd
import Config
from Util import CSVManager
from enum import Enum

'''
데이터 추출 시 Model데이터를 추출할지 , 고정 TOD 결과 데이터를 추출할지 선택하는 열거 클래스
'''
class DataMode(Enum):
    MODEL = 1
    FIX = 2

'''
_dataMode에 따라 PredictID나 FixedID 입력
'''
def main(_dataMode:DataMode, _predictID:int, _fixedID:int):
    _csvMgr = CSVManager()
    if _dataMode == DataMode.MODEL:
        _fileName = 'Model_DataCheck.csv'
    else:
        _fileName = 'Fix_DataCheck.csv'
        
    _csvMgr.Open(Config._basePath,_fileName,['StartTime',
                                                   'Reward',
                                                   'SumPassTime',
                                                   'SumPassVehicleCount',
                                                   'PatternNum',
                                                   'Time',
                                                   'Day',
                                                   'IsNationalDay',
                                                   
                                                   'Link1_MeanSpeed',
                                                   'Link1_MeanPassTime',
                                                   'Link1_SumPassTime',
                                                   'Link1_Cong',
                                                   'Link1_InCount',
                                                   'Link1_QueueCount',
                                                   'Link1_OutCount',
                                                   'Link1_Density',
                                                   
                                                   'Link2_MeanSpeed',
                                                   'Link2_MeanPassTime',
                                                   'Link2_SumPassTime',
                                                   'Link2_Cong',
                                                   'Link2_InCount',
                                                   'Link2_QueueCount',
                                                   'Link2_OutCount',
                                                   'Link2_Density',
                                                   
                                                   'Link3_MeanSpeed',
                                                   'Link3_MeanPassTime',
                                                   'Link3_SumPassTime',
                                                   'Link3_Cong',
                                                   'Link3_InCount',
                                                   'Link3_QueueCount',
                                                   'Link3_OutCount',
                                                   'Link3_Density',
                                                   
                                                   'Link4_MeanSpeed',
                                                   'Link4_MeanPassTime',
                                                   'Link4_SumPassTime',
                                                   'Link4_Cong',
                                                   'Link4_InCount',
                                                   'Link4_QueueCount',
                                                   'Link4_OutCount',
                                                   'Link4_Density',
                                                   ],'w')
    
    _mdb = mdb(Config._data_DBName,
               Config._ip,
               Config._port,
               Config._id,
               Config._pw)
    
    _mdb.Connect()
    if _dataMode == DataMode.MODEL:
        _nData = _mdb.Select(
            f'''SELECT StartTime, Reward, PeriodSumPassTime, 
            SumPassVehicleCount, PatternNum, IntersectionState ,IntersectionStateRaw
            FROM predict_learning_info where predictid = {_predictID};'''
            )
    else : 
        _nData = _mdb.Select(
            f'''SELECT StartTime, Reward, PeriodSumPassTime, 
            SumPassVehicleCount, PatternNum, IntersectionState ,IntersectionStateRaw
            FROM fixed_learning_info where fixedID = {_fixedID};'''
            )
    _mdb.DisConnect()
    
    _test = pd.DataFrame(_nData, columns=['StartTime', 
                                          'Reward', 
                                          'PeriodSumPassTime', 
                                          'SumPassVehicleCount', 
                                          'PatternNum', 
                                          'IntersectionState',
                                          'IntersectionStateRaw'])
   
    _stepList = []
    _linkList = []
    _isInTime = True
    for _valStr in _test['IntersectionState']:
        _linkDensitys = _valStr.split(',')
        _valList = []
        for _index, _vStr in enumerate(_linkDensitys):
            if _index % 3 == 0 and _index != 0:
                _linkList.append(_valList)
                _valList = []
            _valList.append(float(_vStr))
        _linkList.append(_valList)
        _stepList.append(_linkList)
        _linkList = []
        
    _test['ParsedIntersectionState'] = _stepList

    _linkRawList = []
    _stepRawList = []
    for _valStr in _test['IntersectionStateRaw']:
        _rawData = _valStr.split(',')
        _valList = []
        for _index, _rawStr in enumerate(_rawData):
            if _index % 5 == 0 and _index != 0:
                _linkRawList.append(_valList)
                _valList = []
            _valList.append(float(_rawStr))
        _linkRawList.append(_valList)
        _stepRawList.append(_linkRawList)
        _linkRawList = []
    
    _test['ParsedIntersectionStateRaw'] = _stepRawList

    for _idx, _item in _test.iterrows():
        _item["StartTime"] = str(_item["StartTime"])
        _strList = []
        _timeStrList = _item["StartTime"].split(' ')
        _timeList = _timeStrList[2].split(':')
        _item["StartTime"] = dt.timedelta(hours=int(_timeList[0]),minutes=int(_timeList[1]),seconds=int(_timeList[2])).total_seconds()
        print(f'동작시간 : {int(_item["StartTime"])}, Reward : {_item["Reward"]}, 통과시간 합 : {_item["PeriodSumPassTime"]}, 통과차량 : {_item["SumPassVehicleCount"]}, 패턴번호 {_item["PatternNum"]}')
        
        _pared = _item['ParsedIntersectionState']
        _rawPared = _item['ParsedIntersectionStateRaw']
        _strList.extend([_item["StartTime"],_item["Reward"],_item["PeriodSumPassTime"],_item["SumPassVehicleCount"],_item["PatternNum"]])
        if len(_pared) == 5:
            # 시간, 요일, 특수요일 없음
            for _index, _val in enumerate(_pared):
                if _index == 0:
                    #print(f'해당시간 : {_val[0]}, 요일 : {_val[1]}, 특수요일 : {_val[2]}')
                    _strList.extend([_val[0],_val[1],_val[2]])
                else:
                    #print(f'Link{_index}  평균속도 : {_val[0]:.2f}, \t평균통과시간 : {_val[1]:.2f}, \t밀도차이 : {_val[2]:.2f}')
                    _strList.extend([_val[0],
                                     _val[1],
                                     _rawPared[_index-1][4],
                                     _val[2],
                                     _rawPared[_index-1][0],
                                     _rawPared[_index-1][1],
                                     _rawPared[_index-1][2],
                                     _rawPared[_index-1][3]])
        elif len(_pared) == 4:
            # 시간, 요일, 특수요일 있음
            for _index, _val in enumerate(_pared):
                print(f'Link{_index + 1}  평균속도 : {_val[0]:.2f}, \t평균통과시간 : {_val[1]:.2f}, \t밀도차이 : {_val[2]:.2f}')
                _strList.extend([_val[0], _val[1], _val[2]])
        _csvMgr.Write(_strList)
    _csvMgr.Close()
    print()

if __name__ == "__main__":
    main(DataMode.MODEL,607,48)