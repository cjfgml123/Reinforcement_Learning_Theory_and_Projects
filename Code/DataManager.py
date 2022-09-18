
from datetime import datetime
from mariaDBSql import MariaDBSql as mdbSql
import Config
import pandas as pd
import numpy as np

class DataManager:
    def __init__(self):
        self.__mdb: mdbSql = None

    def Initialize(self):
        self.__mdb = mdbSql(Config._info_DBName, Config._ip, Config._port, Config._id, Config._pw)
        self.__mdb.Connect()

    def Finalize(self):
        if self.__mdb is not None:
            self.__mdb.DisConnect()
    
    def GetIntersectionID(self, _nodeID: str):
        _InterData = self.__mdb.Select(
            f'SELECT IntersectionID FROM intersection_info WHERE NODEID = "{_nodeID}";')
        return _InterData[0][0]

    def GetInLinkID(self, _intersectionID: int):
        _LinkDatas = self.__mdb.Select(
            f'SELECT LinkID FROM link_info WHERE EndIntersectionID = {_intersectionID};')
        return [_link[0] for _link in _LinkDatas]

    def GetOutLinkID(self, _intersectionID: int):
        _LinkDatas = self.__mdb.Select(
            f'SELECT LinkID FROM link_info WHERE StartIntersectionID = {_intersectionID};')
        return [_link[0] for _link in _LinkDatas]

    def GetTODID(self, _intersectionID: int):
        _todData = self.__mdb.Select(
            f'SELECT TODID FROM tod WHERE IntersectionID = {_intersectionID};')
        return _todData[0][0]

    def GetTODDatas(self, _intersectionID: int):
        _todData = self.__mdb.Select(
            f'SELECT * FROM tod WHERE IntersectionID = {_intersectionID};')
        return _todData

    def GetTODPlanID(self, _todID: int):
        _planData = self.__mdb.Select(
            f'SELECT planID FROM tod_plan_info WHERE TODID = {_todID};')
        return _planData

    def GetTODPlans(self, _planIDs):
        _inStr = ','.join([str(_id[0]) for _id in _planIDs])
        _planData = self.__mdb.Select(
            f'SELECT PlanID,Number,PatternNum,HourTime FROM tod_plan_detailinfo WHERE PlanID in ({_inStr});')
        return _planData
    
    def GetTODPatterns(self, _todID: int):
        _patternDatas = self.__mdb.Select(
            f'SELECT PatternID,PatternNum,PhaseSequence,PhasePeriod FROM tod_pattern_info WHERE TODID = {_todID};')
        return _patternDatas

    def GetTODPhaseAction(self, _todID: int):
        _phaseActionDatas = self.__mdb.Select(
            f'SELECT PHASENUM,LANEIDSTR FROM phase_Act WHERE todid = {_todID};')
        _phaseActions = []
        for _index, _onePhase in enumerate(_phaseActionDatas):
            _listAction = _onePhase[1].split('-')
            _phaseActions.append([_onePhase[0], []])
            for _laneID in _listAction:
                _phaseActions[_index][1].append(_laneID)
        return _phaseActions

    def GetPhaseLanes(self, _phaseAct: list):
        _lanes = []
        for _a in _phaseAct:
            _jstr = ','.join(str(_b) for _b in _a[1])
            _lanes.append([list(_q) for _q in self.__mdb.Select(
                f'SELECT LANEID, (LANENUM - 1100) FROM LANE_INFO WHERE LANEID IN ({_jstr});')])
            _lanes[-1].insert(0, _a[0])
        return _lanes

    def GetLnae(self, _linkID: str):
        #_test = _mdb.Select(f'SELECT LANENUM FROM LANE_INFO WHERE SUBSTR(CAST(LANENUM AS CHAR(5)),1,1) = "1" AND SUBSTR(CAST(LANENUM AS CHAR(5)),2,1) = "1" AND LINKID = "_linkID";')
        _laneDatas = self.__mdb.Select(
            f'SELECT LANENUM FROM LANE_INFO WHERE (LANENUM-1100) < 100 AND LINKID = "{_linkID}";')
        return _laneDatas

class VDSDataManager:
    def __init__(self) -> None:
        self.__mdb: mdbSql = None
        pass
    
    def Initialize(self):
        self.__mdb = mdbSql(Config._info_DBName, Config._ip, Config._port, Config._id, Config._pw)
        self.__mdb.Connect()

    def Finalize(self):
        if self.__mdb is not None:
            self.__mdb.DisConnect()

    def GetVDSDataFifteen(self):
        _data = pd.read_csv(Config._vdsFilePath)
        _setData = _data.groupby(['TOT_DT'])
        return _setData
    
    def GetVDSDataDay(self):
        _data = pd.read_csv(Config._vdsFilePath)
        _data[['DAY', 'TIME']] = _data['TOT_DT'].str.split(' ',n=1, expand=True)
        #_data['DAY'] = _data['TOT_DT'].str.split(' ',n=1)
        _setData = _data.groupby(['DAY'])
        return _setData
    
    def GetVDSData(self, _nodeID, _date):
        _query = f'SELECT TOT_DT, AVG_SPD, ACSR_ID, TRF_QNTY, DRCT_CD FROM vds_spin_traffic_data WHERE NodeID = "{_nodeID}" and DATE_FORMAT(TOT_DT,"%Y-%m-%d") = "{_date}";'
        _data = self.__mdb.Select(_query)
        _df = pd.DataFrame(_data, columns=['TOT_DT', 'AVG_SPD', 'ACSR_ID', 'TRF_QNTY', 'DRCT_CD'])
        _df['DAY'] = _df['TOT_DT'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))
        _df['TIME'] = _df['TOT_DT'].apply(lambda x: datetime.strftime(x, '%H:%M:%S'))
        return _df.groupby(['DAY'])
    
    def InsertSpinData(self, _path):
        _data = pd.read_csv(_path)
        for _index, _row in _data.iterrows():
            _query = f'''INSERT INTO vds_spin_traffic_data(NodeID,TOT_DT,AVG_SPD,LOS,ACSR_ID,TRF_QNTY,DRCT_CD) 
            VALUES("{_row.NODE_ID}","{_row.TOT_DT}",{_row.AVG_SPD},{_row.LOS},"{_row.ACSR_ID}",{_row.TRF_QNTY},{_row.DRCT_CD});'''
            self.__mdb.Insert(_query)
        self.__mdb.Commit()

if __name__ == '__main__':
    from RL_RunClass import RLPredict
    from DQNManager import PredictMode
    _qq = VDSDataManager()
    _qq.Initialize()
    _predictSimul = RLPredict('2710228400', '월곡사거리', 15, PredictMode.BEST, '2022-01-18')
    _predictSimul.Run()
    _qq.Finalize()