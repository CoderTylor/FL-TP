import pandas as pd
import numpy as np
def sortData(csv_file):
    headerFlag=0
    dataS = pd.read_csv(csv_file)
    max_vehiclenum = np.max(dataS.vehicle_ID.unique())
    for vid in dataS.vehicle_ID.unique():
        # print('{0} and {1}'.format(vid, max_vehiclenum))
        frame_ori = dataS[dataS.vehicle_ID == vid]
        frame_ori.sort_values("time", inplace=True)
        if headerFlag==0:
            frame_ori.to_csv('allData.csv', mode='a', index=False, header=True)
            headerFlag=1
        else:
            frame_ori.to_csv('allData.csv', mode='a', index=False, header=False)
        # print(frame_ori)


def main():
    sortData("testresult.csv")
if __name__ == '__main__':
    main()