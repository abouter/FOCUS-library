import glob
import spe_loader as sl
import pandas as pd
import numpy as np


def LoadData():
    # Compute wavelength
    a = 2.354381287460651
    b = 490.05901104995587
    PixelNumber = np.linspace(1, 1024, 1024)
    CenterPixel = 750
    Wavelength = (PixelNumber-b)/a+CenterPixel

    Folder = glob.glob('./Mes*')
    CycleStore = pd.DataFrame()
    DataTot = []
    for j in range(len(Folder)):
        File = glob.glob(Folder[j]+'/*spe')
        DataRaw = sl.load_from_files(File)
        MetaData = pd.DataFrame(DataRaw.metadata)
        TimeI = MetaData.loc[:, 0].to_numpy()/(1E6)  # Time in ms

        DataTotTemp = pd.DataFrame(np.squeeze(
            DataRaw.data[:][:]), columns=Wavelength)
        DataTotTemp['Mes'] = j
        DataTotTemp['Time'] = TimeI
        DataTot.append(DataTotTemp)

        FileCycle = pd.read_csv(Folder[j]+'\Cycle.csv')
        CycleStore = pd.concat([CycleStore, FileCycle], axis=1)

    DataTot = pd.concat(DataTot).set_index(['Mes', 'Time'])
    return DataTot, CycleStore


if __name__ == '__main__':

    Data, CycleStore = LoadData()
    print('Finished loading, beginning compression')
    Data.to_pickle("./TimeTraceFull.pkl", compression='xz')
    CycleStore.to_csv('./BatchCycle.csv', index=False)
