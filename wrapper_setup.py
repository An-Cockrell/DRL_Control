# compile all .cpp files with the collowing command
# g++ -fPIC -Wall -shared -o Simulation.so ./CPP/*.cpp -std=c++11 -O30


import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

def setUpWrapper():
    sim = ctypes.CDLL('./Simulation.so')

    sim.CreateInstance.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_float))
    sim.CreateInstance.restype = ctypes.POINTER(ctypes.c_void_p)
    sim.CopyInstance.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.CopyInstance.restype = ctypes.POINTER(ctypes.c_void_p)
    sim.singleStep.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.giveABX.argtypes = (ctypes.POINTER(ctypes.c_void_p),)


    sim.getSimulationStep.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getOxydef.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getOxydef.restype = ctypes.c_float
    sim.getSystemOxy.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getSystemOxy.restype = ctypes.c_float
    sim.getTotalInfection.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotalInfection.restype = ctypes.c_float
    sim.getTotal_TNF.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_TNF.restype = ctypes.c_float
    sim.getTotal_sTNFr.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_sTNFr.restype = ctypes.c_float
    sim.getTotal_IL10.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_IL10.restype = ctypes.c_float
    sim.getTotal_GCSF.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_GCSF.restype = ctypes.c_float
    sim.getTotal_proTH1.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_proTH1.restype = ctypes.c_float
    sim.getTotal_proTH2.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_proTH2.restype = ctypes.c_float
    sim.getTotal_IFNg.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_IFNg.restype = ctypes.c_float
    sim.getTotal_PAF.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_PAF.restype = ctypes.c_float
    sim.getTotal_IL1.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_IL1.restype = ctypes.c_float
    sim.getTotal_IL4.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_IL4.restype = ctypes.c_float
    sim.getTotal_IL8.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_IL8.restype = ctypes.c_float
    sim.getTotal_IL12.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_IL12.restype = ctypes.c_float
    sim.getTotal_sIL1r.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_sIL1r.restype = ctypes.c_float
    sim.getTotal_IL1ra.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTotal_IL1ra.restype = ctypes.c_float
    sim.getPAFmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getPAFmult.restype = ctypes.c_float
    sim.getTNFmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getTNFmult.restype = ctypes.c_float
    sim.getsTNFrmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getsTNFrmult.restype = ctypes.c_float
    sim.getIL1ramult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getIL1ramult.restype = ctypes.c_float
    sim.getsIL1rmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getsIL1rmult.restype = ctypes.c_float
    sim.getIFNgmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getIFNgmult.restype = ctypes.c_float
    sim.getIL1mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getIL1mult.restype = ctypes.c_float
    sim.getIL4mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getIL4mult.restype = ctypes.c_float
    sim.getIL8mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getIL8mult.restype = ctypes.c_float
    sim.getIL10mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getIL10mult.restype = ctypes.c_float
    sim.getIL12mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getIL12mult.restype = ctypes.c_float
    sim.getGCSFmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getGCSFmult.restype = ctypes.c_float
    sim.getAllSignalsReturn.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
    sim.getAllSignalsReturn.restype = ndpointer(dtype=ctypes.c_float, shape=(20,10000))

    sim.setSeed.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_int)
    sim.setPAFmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setTNFmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setsTNFrmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setIL1ramult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setsIL1rmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setIFNgmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setIL1mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setIL4mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setIL8mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setIL10mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setIL12mult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)
    sim.setGCSFmult.argtypes = (ctypes.POINTER(ctypes.c_void_p),ctypes.c_float)

    return sim
if __name__ == '__main__':
    setUpWrapper(simObj)
