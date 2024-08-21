USE_DUMMY = True

if USE_DUMMY:
    import ControlDummy as shutter
    import ControlDummy as las
    import ControlDummy as picker
    import ControlDummy as EMCCD
    import ControlDummy as Transla
    import ControlDummy as time
else:
    import ControlFlipMount as shutter
    import ControlLaser as las
    import ControlPulsePicker as picker
    import ControlEMCCD as EMCCD
    import ControlPiezoStage as Transla
    import time as time

import numpy as np
import pandas as pd
import os
import sys
import FileControl

def generateRandomParameters(Nb_points, Nb_Cycle):
    ##############################################################
    # Parameter space and random choice
    ##############################################################

    ###################
    # RNG declaration
    ###################
    rng = np.random.default_rng()

    ###################
    # Proba density function Power
    ###################
    P = (0, 4.4, 10, 100, 800)  # power in uW
    P_calib = (500, 500, 900, 2500, 9200)  # Power from the pp to reach values of P
    #p0 = [0.2, 0.2, 0.2, 0.2, 0.2]
    p1 = [0.3, 0.175, 0.175, 0.175, 0.175]
    ProbaP = p1

    ###################
    # Proba density function Time
    ###################
    t = (0.1, 1, 10, 100)  # time
    #p0 = [0.25, 0.25, 0.25, 0.25]
    p1 = [0.3, 0.23, 0.23, 0.24]
    ProbaT = p1

    df_t_cyc = pd.DataFrame()
    df_p_cyc = pd.DataFrame()
    df_p_cyc_calib = pd.DataFrame()

    for k in range(Nb_Points):
        # Intensity/Power Cycle generation
        df_t_cyc[k] = rng.choice(t, Nb_Cycle, p=ProbaT)
        # First we generate an array of cycle which only contains index for the moment
        temp = rng.choice(np.linspace(0, len(P), len(P), endpoint=False, dtype=int), Nb_Cycle, p=ProbaP)
        while temp[0] == 0:  # We assume that the first element of P is the zero power element
            temp = rng.choice(np.linspace(0, len(P), len(P), endpoint=False, dtype=int), Nb_Cycle, p=ProbaP)
        df_p_cyc_calib[k] = np.array([P_calib[i] for i in temp])
        df_p_cyc[k] = np.array([P[i] for i in temp])

    return df_t_cyc, df_p_cyc, df_p_cyc_calib
    

class timeTraceRunner:
    def __init__(self, **kwargs):
        self.GeneralPara = kwargs
        if 'Nb_points' not in self.GeneralPara:
            raise ValueError('Parameters must contain the number of points [\'Nb_points\']')
        self.Nb_Points = GeneralPara['Nb_points']

    def initialize(self, start_x, end_x, start_y, end_y):
        runner.initializePiezo(start_x, end_x, start_y, end_y)
        runner.initializeInstruments()
        runner.initializeConex()
        runner.initializeOutputDirectory()

    #############################
    # Piezo parameter
    #############################
    def initializePiezo(self, start_x, end_x, start_y, end_y):
        x = np.linspace(start_x, end_x, int(np.floor(np.sqrt(self.Nb_Points))))
        y = np.linspace(start_y, end_y, int(np.ceil(np.sqrt(self.Nb_Points))))
        X, Y = np.meshgrid(x, y)
        self.Pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
        try:
            print('Number of Points:{}\nDistance between points:\n\t x ={} \n\t y ={}'.format(len(self.Pos),x[1]-x[0], y[1]-y[0]))
        except IndexError:
            print('Number of Points:{}\n'.format(len(self.Pos)))

    def initializeInstruments(self):
        self.InstrumentsPara = {}

        #############################
        # Initialisation of laser
        #############################
        self.Laser = las.LaserControl('COM8', 'COM17', 0.5)
        self.InstrumentsPara['Laser'] = self.Laser.parameterDict
        print('Initialised Laser')

        #############################
        # Initialisation of pulse picker
        #############################
        self.pp = picker.PulsePicker("USB0::0x0403::0xC434::S09748-10A7::INSTR")
        self.InstrumentsPara['Pulse picker'] = self.pp.parameterDict
        print('Initialised pulse picker')

        #############################
        # Initialisation of the EMCCD
        #############################
        self.camera = EMCCD.LightFieldControl('TimeTraceEM')
        #FrameTime = camera.GetFrameTime()
        #ExposureTime = camera.GetExposureTime()
        #NumberOfFrame = camera.GetNumberOfFrame()
        self.InstrumentsPara['PI EMCCD'] = self.camera.parameterDict
        print('Initialised EMCCD')

        #############################
        # Initialisation of the shutter
        #############################
        self.FM = shutter.FlipMount("37007726")
        print('Initialised Flip mount')

    #############################
    # Initialisation of the Conex Controller
    #############################
    def initializeConex(self):
        if 'ControlConex' in sys.modules:
            self.x_axis = Transla.ConexController('COM12')
            self.y_axis = Transla.ConexController('COM13')
            print('Initialised rough translation stage')
        elif 'ControlPiezoStage' in sys.modules or USE_DUMMY:
            piezo = Transla.PiezoControl('COM15')
            self.x_axis = Transla.PiezoAxisControl(piezo, 'x')
            self.y_axis = Transla.PiezoAxisControl(piezo, 'y')
            print('Initialised piezo translation stage')

    #############################
    # Preparation of the directory
    #############################
    def initializeOutputDirectory(self):
        print('Directory staging, please check other window')
        out_dir = None
        if USE_DUMMY:
            out_dir = 'output-dummy'
            if(not os.path.isdir(out_dir)):
                os.makedirs(out_dir)
        self.DirectoryPath = FileControl.PrepareDirectory(self.GeneralPara, self.InstrumentsPara, out_dir)

    #############################
    # TimeTrace loop
    #############################
    def runTimeTrace(self, StabilityTime, df_t_cyc, df_p_cyc, df_p_cyc_calib):
        print('')
        MesNumber = np.linspace(0, self.Nb_Points, self.Nb_Points, endpoint=False)
        IteratorMes = np.nditer(MesNumber, flags=['f_index'])

        Nb_Cycle = len(df_t_cyc[0])
        CycNumber = np.linspace(0, Nb_Cycle, Nb_Cycle, endpoint=False)
        IteratorCyc = np.nditer(CycNumber, flags=['f_index'])

        self.Laser.SetStatusShutterTunable(1)
        self.FM.ChangeState(0)
        for k in IteratorMes:
            # Generation of the folder and measurement prep
            print('Measurement number:{}'.format(MesNumber[IteratorMes.index]))
            TempDirPath = self.DirectoryPath+'/Mes'+str(MesNumber[IteratorMes.index])+'x='+str(np.round(
                self.Pos[IteratorMes.index, 0], 2))+'y='+str(np.round(self.Pos[IteratorMes.index, 1], 2))
            if(not os.path.isdir(TempDirPath)):
                os.mkdir(TempDirPath)

            self.camera.SetSaveDirectory(TempDirPath.replace('/',"\\"))
            
            self.x_axis.MoveTo(self.Pos[IteratorMes.index, 0])
            self.y_axis.MoveTo(self.Pos[IteratorMes.index, 1])

            # Intensity/Power Cycles
            t_cyc = df_t_cyc[k]
            p_cyc = df_p_cyc[k]
            p_cyc_calib = df_p_cyc_calib[k]
            assert(len(t_cyc) == len(p_cyc) == len(p_cyc_calib) == Nb_Cycle)

            T_tot = np.sum(t_cyc)

            # Camera setting adjustement
            NbFrameCycle = np.ceil((T_tot+StabilityTime)/self.camera.GetFrameTime())
            self.camera.SetNumberOfFrame(NbFrameCycle)
            print('Time cycle:{}'.format(t_cyc.tolist()))
            print('Power cycle:{}'.format(p_cyc.tolist()))
            print('Real Power cycle:{}'.format(p_cyc_calib.tolist()))
            print('Total time={}'.format(T_tot+StabilityTime))

            #Create timing parameter
            t_sync=np.zeros(len(t_cyc))
            self.FM.ChangeState(0)
            self.camera.Acquire()  # Launch acquisition
            t0=time.time()
            # Power iteration
            for j in IteratorCyc:
                print('Cycle {}: P={},t={}'.format(IteratorCyc.index,p_cyc[IteratorCyc.index],t_cyc[IteratorCyc.index]))
                if p_cyc[IteratorCyc.index] == 0:
                    self.FM.ChangeState(0)
                elif p_cyc[IteratorCyc.index] != 0:
                    self.FM.ChangeState(1)
                    self.pp.SetPower(p_cyc_calib[IteratorCyc.index])
                t_sync[IteratorCyc.index]=time.time()-t0
                time.sleep(t_cyc[IteratorCyc.index])
            IteratorCyc.reset()
            # once it finished we set the power to the minimum and continue measurement
            print('Stability Time')
            self.FM.ChangeState(1)
            self.pp.SetPower(np.min(p_cyc_calib))
            self.camera.WaitForAcq()
            self.FM.ChangeState(0)

            # Save all the cycle in the folder
            temp = pd.DataFrame(
                {'Exposure Time': t_cyc.tolist(), 'Power send': p_cyc.tolist(), 'Power Pulse-picker': p_cyc_calib.tolist(),'Sync':t_sync})
            temp.to_csv(TempDirPath+'/Cycle.csv')

        self.Laser.SetStatusShutterTunable(0)


if __name__ == '__main__':

    if not USE_DUMMY:
        os.system('cls')

    #############################
    # Parameters
    #############################

    Nb_Points = 100  # Number of position for the piezo
    Nb_Cycle = 10  # Number of cycle during experiment
    DistancePts = 10
    StabilityTime = 30
    GeneralPara = {'Experiment_name': 'EMCCDRepeatDiffPos', 'Nb_points': Nb_Points,
               'Distance_Between_Points ': DistancePts,
               'Note': 'The SHG unit from Coherent was used'}
    
    start_x = 20
    end_x = 80
    start_y = 20
    end_y = 80
    
    runner = timeTraceRunner(**GeneralPara)
    runner.initialize(start_x, end_x, start_y, end_y)
    df_t_cyc, df_p_cyc, df_p_cyc_calib = generateRandomParameters(Nb_Points, Nb_Cycle)
    runner.runTimeTrace(StabilityTime, df_t_cyc, df_p_cyc, df_p_cyc_calib)

