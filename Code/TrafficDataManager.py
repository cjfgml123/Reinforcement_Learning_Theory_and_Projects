from mariaDBSql import MariaDBSql as mdb

class TraffciDataManager:
    def __init__(self,_dbName:str,_ip:str,_port:int,_id:str,_pw:str):
        self.__dbMgr = mdb(_dbName,_ip,_port,_id,_pw)
        self.__dbMgr.Connect()
        
    def __del__(self):
        self.__dbMgr.DisConnect()
    
    def Commit(self):
        self.__dbMgr.Commit()
    
    def RollBack(self):
        self.__dbMgr.RollBack()
    
    def SelectModel(self, _modelName:str):
        _query = f'SELECT ModelID FROM MODEL_INFO WHERE ModelName="{_modelName}";'
        _selData = self.__dbMgr.Select(_query)
        #if len(_selData[0]) != 0:
        if len(_selData) != 0:
            return _selData[0][0]
        return None

    def InsertModel(self, _modelName:str,
                    _inputSize, _outputSize, _bufferSize,
                    _learningRate, _batchSize, _updatePeriod,
                    _gamma, _trainSize, _epsilonStart, _epsiloneEnd,
                    _episodeCount):
        # ModelID, InputSize, OutputSize, BufferSize, LearningRate, BatchSize, UpdatePeriod, Gamma, TrainSize, EpsilonStart, EpsilonEnd, EpisodeCount, ModelName
        _query = f'''INSERT MODEL_INFO(ModelName,InputSize,OutputSize,BufferSize,LearningRate,BatchSize,UpdatePeriod,
        Gamma,TrainSize,EpsilonStart,EpsilonEnd,EpisodeCount) 
        VALUES("{_modelName}",{_inputSize},{_outputSize},{_bufferSize},{_learningRate},{_batchSize},{_updatePeriod},
        {_gamma},{_trainSize},{_epsilonStart},{_epsiloneEnd},{_episodeCount});'''
        
        self.__dbMgr.Insert(_query)
        self.__dbMgr.Commit()
        _query = 'SELECT last_insert_id()'
        # [()] 리턴
        _selData = self.__dbMgr.Select(_query)
        _id = _selData[0][0]
        return _id
    
    def UpdateModel(self,_modelID:int,_inputSize, _outputSize, _bufferSize,
                    _learningRate, _batchSize, _updatePeriod,
                    _gamma, _trainSize, _epsilonStart, _epsilonEnd,
                    _episodeCount):
        _query = f'''UPDATE MODEL_INFO SET InputSize = {_inputSize}, OutputSize = {_outputSize}, BufferSize = {_bufferSize},
        LearningRate = {_learningRate}, BatchSize = {_batchSize}, UpdatePeriod = {_updatePeriod}, Gamma = {_gamma}, TrainSize = {_trainSize},
        EpsilonStart = {_epsilonStart}, EpsilonEnd = {_epsilonEnd}, EpisodeCount = {_episodeCount} WHERE ModelID = {_modelID};'''
        self.__dbMgr.Insert(_query)
        self.__dbMgr.Commit()
        
    def SelectHistory(self,_modelID):
        _query = f'''SELECT HistoryID FROM MODEL_HISTORY_INFO WHERE ModelID={_modelID};'''
        _selData = self.__dbMgr.Select(_query)
        if len(_selData) != 0:
            return _selData[len(_selData)-1][0]
        return None
    
    def SelectProgressedVals(self,_historyID):
        _query = f'''SELECT ProgressedEpisode,ValAvgPassTime,ValAvgReward FROM MODEL_HISTORY_INFO WHERE HistoryID= {_historyID};'''
        _selData = self.__dbMgr.Select(_query)
        if (len(_selData) != 2):
            return _selData[0][0],_selData[0][1],_selData[0][2]
        return None
    
    def InsertHistory(self,_modelID,_startDate:str):
        # HistoryID,ModelID,StartDate
        _query = f'''INSERT MODEL_HISTORY_INFO(ModelID,StartDate)
        VALUES({_modelID},"{_startDate}");'''
        self.__dbMgr.Insert(_query)
        self.__dbMgr.Commit()
        _query = 'SELECT last_insert_id()'
        # [()] 리턴
        _selData = self.__dbMgr.Select(_query)
        _id = _selData[0][0]
        return _id
    
    def EveryUpdateHistory(self,_historyID,_progressedEpisode,_progressedEpsilon):
        _query = f'''UPDATE MODEL_HISTORY_INFO SET ProgressedEpisode =  {_progressedEpisode}, 
        ProgressedEpsilon = {_progressedEpsilon} WHERE HistoryID = {_historyID};'''
        self.__dbMgr.Update(_query)
        self.__dbMgr.Commit()
        
    def PeriodUpdateHistory(self,_historyID,_bestReward):
        _query = f'''UPDATE MODEL_HISTORY_INFO SET BestReward = {_bestReward} WHERE HistoryID = {_historyID};'''
        self.__dbMgr.Update(_query)
        self.__dbMgr.Commit()

    def OneUpdateHistory(self,_historyID,_endTime:str,_historyState,_errorMsg):
        _query = f'''UPDATE MODEL_HISTORY_INFO SET EndDate = '{_endTime}', State = '{_historyState}', ErrorMsg = '{_errorMsg}' WHERE HistoryID = {_historyID};'''
        self.__dbMgr.Update(_query)
        self.__dbMgr.Commit()

    # def OneUpdateHistory(self,_historyID,_endTime:str,_historyState,_errorMsg):
    #     _query = f'''UPDATE MODEL_HISTORY_INFO SET EndDate = '{_endTime}' WHERE HistoryID = {_historyID};'''
    #     self.__dbMgr.Update(_query)
    #     self.__dbMgr.Commit()
    
    def UpdateHistoryValData(self,_historyID:int,_avgPassTime:float,_avgReward:float,_bestEpi:int):
        _query = f'''UPDATE MODEL_HISTORY_INFO SET ValAvgPassTime = {_avgPassTime}, ValAvgReward = {_avgReward}, BestEpisode = {_bestEpi}  WHERE HistoryID = {_historyID}'''
        self.__dbMgr.Update(_query)
        self.__dbMgr.Commit()
    
    def InsertEpisode(self,_historyID,_EpisodeNum,_Epsilon):
        # EpisodeID,HistoryID,EpisodeNum,Epsilon
        _query = f'''INSERT IGNORE EPISODE_LEARNING_INFO(HistoryID,EpisodeNum,Epsilon)
        VALUES({_historyID},{_EpisodeNum},{_Epsilon});'''
        self.__dbMgr.Insert(_query)
        self.__dbMgr.Commit()
        _query = 'SELECT last_insert_id()'
        # [()] 리턴
        _selData = self.__dbMgr.Select(_query)
        _id = _selData[0][0]
        return _id
    
    def UpdateEpisode(self,_episodeID,_accmulatePassTime,_accmulateReward,_bufferSize,_episodeState):
        _query = f'''UPDATE EPISODE_LEARNING_INFO SET AccmulatePassTime = {_accmulatePassTime},
                AccmulateReward = {_accmulateReward}, BufferSize = {_bufferSize}, State = "{_episodeState}" 
                WHERE EpisodeID = {_episodeID};'''
        self.__dbMgr.Update(_query)
    
    def InsertLearnData(self,_episodeID,_startTime:str,_reward,_periodSumPassTime,_sumPassVehicleCount,_patternNum,_IntersectionState:str,_intersectionStateRaw:str,_intersectionID:int,_periodAvgSpeed:float,_vehicleAvgPassTime:float,_vehicleInCountSum:int):
        _query = f'''INSERT learning_info(EpisodeID,StartTime,Reward,PeriodSumPassTime,SumPassVehicleCount,PatternNum,IntersectionState,IntersectionStateRaw,IntersectionID,PeriodAvgSpeed,VehicleAvgPassTime,VehicleInCountSum)
        VALUES({_episodeID},"{_startTime}",{_reward},{_periodSumPassTime},{_sumPassVehicleCount},{_patternNum},
                "{_IntersectionState}","{_intersectionStateRaw}",{_intersectionID},{_periodAvgSpeed},{_vehicleAvgPassTime},{_vehicleInCountSum});'''
        self.__dbMgr.Insert(_query)
    
    def InsertPredictInfo(self,
                          _modelID,
                          _modelType,
                          _dataInfo:str):
        _query = f'''INSERT model_predict_info(ModelID,ModelType,DataInfo)
        VALUES({_modelID},"{_modelType}","{_dataInfo}");'''
        self.__dbMgr.Insert(_query)
        self.__dbMgr.Commit()
        _query = 'SELECT last_insert_id()'
        _selData = self.__dbMgr.Select(_query)
        _id = _selData[0][0]
        return _id
    
    def InsertPredictLearnData(self,
                               _predictID,
                               _startDate,
                               _startTime:str,
                               _reward,
                               _periodSumPassTime,
                               _sumPassVehicleCount,
                               _patternNum,
                               _IntersectionState:str,
                               _intersectionStateRaw:str,
                               _intersectionID:int,
                               _periodAvgSpeed:float,
                               _vehicleAvgPassTime:float,
                               _vehicleInCountSum:int):
        _query = f'''INSERT predict_learning_info(PredictID,StartDate,StartTime,Reward,PeriodSumPassTime,SumPassVehicleCount,PatternNum,IntersectionState,IntersectionStateRaw,IntersectionID,PeriodAvgSpeed,VehicleAvgPassTime,VehicleInCountSum)
            VALUES({_predictID},"{_startDate}","{_startTime}",{_reward},{_periodSumPassTime},{_sumPassVehicleCount},{_patternNum},
            "{_IntersectionState}","{_intersectionStateRaw}",{_intersectionID},{_periodAvgSpeed},{_vehicleAvgPassTime},{_vehicleInCountSum});'''
        self.__dbMgr.Insert(_query)
    
    def UpdatePredictInfo(self,_predictID,_accmulatePassTime,_accmulateReward):
        _query = f'''UPDATE model_predict_info SET AccmulatePassTime = {_accmulatePassTime},
                AccmulateReward = {_accmulateReward} WHERE PredictID = {_predictID};'''
        self.__dbMgr.Update(_query)
    
    def InsertFixedInfo(self, _modelID, _modelType, _dataInfo:str):
        _query = f'''INSERT model_fixed_info(ModelID,ModelType,DataInfo)
        VALUES({_modelID},"{_modelType}","{_dataInfo}");'''
        self.__dbMgr.Insert(_query)
        self.__dbMgr.Commit()
        _query = 'SELECT last_insert_id()'
        _selData = self.__dbMgr.Select(_query)
        _id = _selData[0][0]
        return _id
    
    def InsertFixedLearnData(self,_fixedID,_startDate,_startTime:str,_reward,_periodSumPassTime,_sumPassVehicleCount,_patternNum,_IntersectionState:str,_intersectionStateRaw:str,_intersectionID:int,_periodAvgSpeed:float,_vehicleAvgPassTime:float,_vehicleInCountSum:int):
        _query = f'''INSERT fixed_learning_info(FixedID,StartDate,StartTime,Reward,PeriodSumPassTime,SumPassVehicleCount,PatternNum,IntersectionState,IntersectionStateRaw,IntersectionID,PeriodAvgSpeed,VehicleAvgPassTime,VehicleInCountSum)
        VALUES({_fixedID},"{_startDate}","{_startTime}",{_reward},{_periodSumPassTime},{_sumPassVehicleCount},{_patternNum},
                "{_IntersectionState}","{_intersectionStateRaw}",{_intersectionID},{_periodAvgSpeed},{_vehicleAvgPassTime},{_vehicleInCountSum});'''
        self.__dbMgr.Insert(_query)
    
    def UpdateFixedInfo(self,_fixedID,_accmulatePassTime,_accmulateReward):
        _query = f'''UPDATE model_fixed_info SET AccmulatePassTime = {_accmulatePassTime},
                AccmulateReward = {_accmulateReward} WHERE FixedID = {_fixedID};'''
        self.__dbMgr.Update(_query)
        self.__dbMgr.Commit()
    # def InsertSimulationFixedCycleData(self,_intersectionID:int,_patternID:int,_startDateTime:str,_periodSumPassTime:float,_periodSumPassVehicleCount:int,_periodAvgSpeed:float,_periodSumInflowVehicleCount:int,_vehicleAvgPassTime:float):
    #     _query = f'''INSERT simulation_fixed_cycle_data(IntersectionID,PatternID,StartDateTime,PeriodSumPassTime,PeriodSumPassVehicleCount,PeriodAvgSpeed,PeriodSumInflowVehicleCount,VehicleAvgPassTime)
    #     VALUES({_intersectionID},{_patternID},"{_startDateTime}",{_periodSumPassTime},{_periodSumPassVehicleCount},{_periodAvgSpeed},{_periodSumInflowVehicleCount},{_vehicleAvgPassTime})'''
    #     self.__dbMgr.Insert(_query)
                
if __name__ == '__main__':
    import Config
    #Sample
    _tdm = TraffciDataManager()
    try:
        _id = _tdm.SelectModel(Config._finalModelName)
        if _id is None:
            _id = _tdm.InsertModel(Config._finalModelName,6,6,Config.buffer_limit,Config.learning_rate,Config.batch_size,Config.updatePeriod,Config.gamma,Config.train_size,
                                   Config._eps_start,Config._eps_end,Config._episodes)
            print("M!")
        print(_id)
        _id = _tdm.InsertHistory(_id,'2022-05-10 00:00:00','2022-05-10 01:00:00',10,10,10)
        print(_id)
        _id = _tdm.InsertEpisode(_id, 10,10,10,10,10)
        print(_id)
        _tdm.InsertLearnData(_id, '00:00:00', 10,10,10,10,10)
        _tdm.Commit()
    except:
        _tdm.RollBack()
        print ("ERROR")