import mysql.connector as mdb

class MariaDBSql:
    def __init__(self, _dbName:str, _ip:str, _port:int, _id:str, _pw:str):
        self.__dbName =_dbName
        self.__ip = _ip
        self.__port = _port
        self.__id = _id
        self.__pw = _pw

    def Connect(self):
        self.__db = mdb.connect(
            user=self.__id, password=self.__pw,
            host=self.__ip, port=self.__port,database=self.__dbName)
        self.__db.autocommit = False
        self.__cur = self.__db.cursor()
        self.__isConnected = True

    def DisConnect(self):
        if self.__isConnected:
            #self.__cur.close()
            self.__db.close()
            self.__isConnected = False

    def __Excute(self, _query:str):
        try:
            self.__cur.execute(_query)
        except Exception as _ex:
            print(_ex)

    def Commit(self):
        self.__db.commit()
        #self.__Excute("COMMIT")

    def RollBack(self):
        self.__db.rollback()
        #self.__Excute("ROLLBACK")

    def Insert(self, _query:str):
        self.__Excute(_query)

    def Update(self, _query:str):
        self.__Excute(_query)

    def Delete(self, _query:str):
        self.__Excute(_query)

    def Select(self, _query:str):
        self.__Excute(_query)
        return self.__cur.fetchall()
