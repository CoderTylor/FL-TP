import pandas as pd
import numpy as np

def load_data(csv_file,add_count):
    dataS = pd.read_csv(csv_file)
    max_vehiclenum = np.max(dataS.vehicle_ID.unique())
    tmpmax_vehiclenum=max_vehiclenum
    headerFlag=0
    Good_Data,All_Data,Bad_Data=0,0,0
    for vid in dataS.vehicle_ID.unique():
        print('{0} and {1}'.format(vid, max_vehiclenum))
        frame_ori = dataS[dataS.vehicle_ID == vid]
        if headerFlag==0:
            frame_ori.to_csv("add_moreAttack.csv",mode='a',header=True)
            headerFlag=1
        else:
            frame_ori.to_csv("add_moreAttack.csv",mode='a',header=False)

        if frame_ori.attackerType.isin([0]).all()==False:
            Bad_Data+=1
            Bad_Data+=add_count
            for i in range(add_count):
                tmpmax_vehiclenum += 1
                frame_ori.vehicle_ID=tmpmax_vehiclenum
                frame_ori.to_csv("add_moreAttack.csv", mode='a', header=False)
        else:
            Good_Data+=1
    All_Data=tmpmax_vehiclenum
    return All_Data,Good_Data,Bad_Data


if __name__ == '__main__':
    file_name="data_sortID.csv"
    add_count=3
    AllData,GoodData,BadData=load_data(file_name,add_count)
    print('CurrentAll:{0}, GoodData:{1}, BadData:{2}'.format(AllData, GoodData,BadData))
