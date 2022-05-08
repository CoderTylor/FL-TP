import os
# 遍历文件夹
import json
import pandas as pd
def walkFile(file):
    count_fileDir=0
    HeaderFlag=0
    for root, dirs, files in os.walk(file):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            if "JSON" in f:
                if f =="GroundTruthJSONlog.json":
                    count_fileDir+=1
                else:
                    JsonFilePath=os.path.join(root, f)
                    groundTruthFilePath=os.path.join(root, "GroundTruthJSONlog.json")
                    readJson(JsonFilePath,count_fileDir,groundTruthFilePath,HeaderFlag)
                    HeaderFlag+=1
                    # print(count_fileDir,os.path.join(root, f))
                # print(f)
            # if f.contains(".json"):
            #     print("AAAA",os.path.join(root, f))
        # 遍历所有的文件夹
        # for d in dirs:
        #     print("BBBB",os.path.join(root, d))

def readJson(filePath,count,groundTruthFilePath,HeaderFlag):
    previousJsonLine=""
    # 需要保存的列表
    sendVehilceIDlsit=[]
    sendVehilceRSSIlist=[]
    sendVehilceTimeSteplist=[]
    distanceChange_Xlist=[]
    distanceChange_Ylist=[]
    speedChange_Xlist=[]
    speedChange_Ylist=[]
    sendVehiclePosion_Xlist=[]
    sendVehiclePosion_Ylist=[]
    sendVehicleSpeed_Xlist=[]
    sendVehicleSpeed_Ylist=[]
    AttackList=[]


    with open(filePath, 'r', encoding="utf-8") as rf:
        for jsonstr in rf.readlines():
            # 将josn字符串转化为dict字典ss
            jsonstr = json.loads(jsonstr)
            if 'RSSI' in jsonstr:#这里是说明这条轨迹数据是无线传输accept过来的
                #需要在GroundTruthFile里面找当前的Attack类型\
                messageID=jsonstr["messageID"]
                attackType=findAttackType(messageID,groundTruthFilePath)
                # print(attackType,jsonstr["rcvTime"], jsonstr["pos"], jsonstr["spd"], jsonstr["RSSI"], jsonstr["sender"],
                #       jsonstr["messageID"])
                # print(previousJsonLine["rcvTime"], previousJsonLine["pos"], previousJsonLine["spd"])

                #当前车辆信息
                egoVehiclePosion_X = previousJsonLine["pos"][0]
                egoVehiclePosion_Y = previousJsonLine["pos"][1]
                egoVehicleSpeed_X = previousJsonLine["spd"][0]
                egoVehicleSpeed_Y = previousJsonLine["spd"][1]

                #发送车辆信息
                sendVehiclePosion_X = jsonstr["pos"][0]
                sendVehiclePosion_Y = jsonstr["pos"][1]
                sendVehicleSpeed_X = jsonstr["spd"][0]
                sendVehicleSpeed_Y = jsonstr["spd"][1]
                sendVehilceID=str(str(count)+str(jsonstr["sender"]))
                sendVehilceRSSI=jsonstr["RSSI"]
                sendVehilceTimeStep=jsonstr["rcvTime"]

                #计算需要保存的信息
                distanceChange_X=abs(sendVehiclePosion_X-egoVehiclePosion_X)
                distanceChange_Y=abs(sendVehiclePosion_Y-egoVehiclePosion_Y)
                speedChange_X=abs(sendVehicleSpeed_X-egoVehicleSpeed_X)
                speedChange_Y=abs(sendVehicleSpeed_Y-egoVehicleSpeed_Y)

                # list append
                sendVehilceIDlsit.append(sendVehilceID)
                sendVehilceRSSIlist.append(sendVehilceRSSI)
                sendVehilceTimeSteplist.append(sendVehilceTimeStep)
                distanceChange_Xlist.append(distanceChange_X)
                distanceChange_Ylist.append(distanceChange_Y)
                speedChange_Xlist.append(speedChange_X)
                speedChange_Ylist.append(speedChange_Y)
                sendVehiclePosion_Xlist.append(sendVehiclePosion_X)
                sendVehiclePosion_Ylist.append(sendVehiclePosion_Y)
                sendVehicleSpeed_Xlist.append(sendVehicleSpeed_X)
                sendVehicleSpeed_Ylist.append(sendVehicleSpeed_Y)
                AttackList.append(attackType)
            previousJsonLine = jsonstr  # 记录当前车辆的位置

    #存csv，pandas
    df = pd.DataFrame({"vehicle_ID": sendVehilceIDlsit, "attackerType": AttackList, "time": sendVehilceTimeSteplist,
                       "pos_x":sendVehiclePosion_Xlist,"pos_y":sendVehiclePosion_Ylist,"spd_x":sendVehicleSpeed_Xlist
                          ,"spd_y":sendVehicleSpeed_Ylist,"disChange_X":distanceChange_Xlist,
                       "disChange_Y":distanceChange_Ylist, "spdChange_X":speedChange_Xlist, "spdChange_Y":speedChange_Ylist,
                       "RSSI":sendVehilceRSSIlist})
    if HeaderFlag==0:
        df.to_csv('result.csv',mode='a', index=False,header=True)
    else:
        df.to_csv('result.csv',mode='a', index=False,header=False)
    print(filePath)

    # print('RSSI' in jsonstr)


    #     data = json.load(rf)
    # return data
def findAttackType(messageID,groundTruthFilePath):
    with open(groundTruthFilePath, 'r', encoding="utf-8") as rf:
        for jsonstr in rf.readlines():
            # 将josn字符串转化为dict字典ss
            jsonstr = json.loads(jsonstr)
            if messageID==jsonstr["messageID"]:
                return jsonstr["attackerType"]


def main():
    walkFile("Generate_Dataset")
if __name__ == '__main__':
    main()