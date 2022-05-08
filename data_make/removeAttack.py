import pandas as pd

def removeAttack(csv_file):
    dataS = pd.read_csv(csv_file)
    dataS = dataS[(dataS['attackerType'] == 0)]
    print(dataS)
    dataS.to_csv('removeAttack_SingleVehicle.csv', mode='a', index=False, header=False)

    # max_vehiclenum = np.max(dataS.vehicle_ID.unique())
    # for vid in dataS.vehicle_ID.unique():
    #     # print('{0} and {1}'.format(vid, max_vehiclenum))
    #     frame_ori = dataS[dataS.vehicle_ID == vid]
    #     # frame_ori.sort_values("time", inplace=True)
    #     if headerFlag==0:
    #         frame_ori.to_csv('allData.csv', mode='a', index=False, header=True)
    #         headerFlag=1
    #     else:
    #         frame_ori.to_csv('allData.csv', mode='a', index=False, header=False)
    #     # print(frame_ori)


def main():
    removeAttack("allData.csv")
if __name__ == '__main__':
    main()