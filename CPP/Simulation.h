//
// Created by Dale Larie on 7/22/20.
//
#include <vector>
#include <random>
#include <stdlib.h>
#include <algorithm>
#include "Parameters.h"
#ifndef IIRABM_C_OBJECT_SIMULATION_H
#define IIRABM_C_OBJECT_SIMULATION_H


mt19937 generator;
uniform_int_distribution<int> distribution10k(0,9999);
uniform_int_distribution<int> distribution1000(0,999);
uniform_int_distribution<int> distribution100(0,99);
uniform_int_distribution<int> distribution50(0,49);
uniform_int_distribution<int> distribution12(0,11);
uniform_int_distribution<int> distribution10(0,9);
uniform_int_distribution<int> distribution9(0,8);
uniform_int_distribution<int> distribution8(0,7);
uniform_int_distribution<int> distribution5(0,4);
uniform_int_distribution<int> distribution3(0,2);
uniform_int_distribution<int> distribution2(0,1);


class SimulationObject {
public:
    SimulationObject(){};
    SimulationObject(float OH, int IS, int NRI, int NIR, int InjNum, int inputSeed, int numCyto, float* IP);
    SimulationObject(const SimulationObject& originalSim);
    void initialize();
    void diffuse();
    void evaporate();
    void updateSystemOxy(int step);
    void injure_infectionFRD(int inj_number);
    void recur_injury();
    void clearIntervention();
    void simulationStep();
    void cellStep();
    void recurrentInjury();
    void giveABX();
    void updateTrajectoryOutput(int q);
    void singleStep();


    vector<EC> ecArray;
    vector<int> ecIndexes;
    vector<pmn> pmnArray;
    vector<mono> monoArray;
    vector<TH0> TH0array;
    vector<TH1> TH1array;
    vector<TH2> TH2array;
    vector<pmn_marrow> pmn_marrowArray;
    vector<mono_marrow> mono_marrowArray;
    vector<TH0_germ> TH0_germArray;
    vector<TH1_germ> TH1_germArray;
    vector<TH2_germ> TH2_germArray;
    float system_oxy,oxyDeficit,totalInfection,total_TNF,total_sTNFr,total_IL10,total_GCSF,total_proTH1,total_proTH2;
    float total_IFNg, total_PAF, total_IL1, total_IL4, total_IL8, total_IL12, total_sIL1r, total_IL1ra;
    float PAFmult, TNFmult, sTNFrmult, IL1ramult, sIL1rmult, IFNgmult, IL1mult, IL4mult, IL8mult, IL10mult, IL12mult,
            GCSFmult;
    int cellGrid[101][101];
    float allSignals[20][numTimeSteps];
    float allSignalsReturn[20*numTimeSteps];
    //    Variables for functions
    float oxyHeal;
    int infectSpread, numRecurInj, numInfectRepeat, inj_number, seed, numCytokines;
    float* internalParameterization;
    int i,step,iend,jend,dOut,count6hr,antibiotic1,antibiotic2,istep,x1,x2,k,j;
    int numABX;
    int flag;

// Getters for variables
    int getStep(){return istep;}
    float getOxydef(){return oxyDeficit;}
    float getSystemOxy(){return system_oxy;}
    float getTotalInfection(){return totalInfection;}
    float getTotal_TNF(){return total_TNF;}
    float getTotal_sTNFr(){return total_sTNFr;}
    float getTotal_IL10(){return total_IL10;}
    float getTotal_GCSF(){return total_GCSF;}
    float getTotal_proTH1(){return total_proTH1;}
    float getTotal_proTH2(){return total_proTH2;}
    float getTotal_IFNg(){return total_IFNg;}
    float getTotal_PAF(){return total_PAF;}
    float getTotal_IL1(){return total_IL1;}
    float getTotal_IL4(){return total_IL4;}
    float getTotal_IL8(){return total_IL8;}
    float getTotal_IL12(){return total_IL12;}
    float getTotal_sIL1r(){return total_sIL1r;}
    float getTotal_IL1ra(){return total_IL1ra;}
    float getPAFmult(){return PAFmult;}
    float getTNFmult(){return TNFmult;}
    float getsTNFrmult(){return sTNFrmult;}
    float getIL1ramult(){return IL1ramult;}
    float getsIL1rmult(){return sIL1rmult;}
    float getIFNgmult(){return IFNgmult;}
    float getIL1mult(){return IL1mult;}
    float getIL4mult(){return IL4mult;}
    float getIL8mult(){return IL8mult;}
    float getIL10mult(){return IL10mult;}
    float getIL12mult(){return IL12mult;}
    float getGCSFmult(){return GCSFmult;}
    float* getAllSignalsReturn();

// setters
    void setSeed(int newSeed){seed = newSeed; generator.seed(seed);}
    void setPAFmult(float newMult){PAFmult = newMult;}
    void setTNFmult(float newMult){TNFmult = newMult;}
    void setsTNFrmult(float newMult){sTNFrmult = newMult;}
    void setIL1ramult(float newMult){IL1ramult = newMult;}
    void setsIL1rmult(float newMult){sIL1rmult = newMult;}
    void setIFNgmult(float newMult){IFNgmult = newMult;}
    void setIL1mult(float newMult){IL1mult = newMult;}
    void setIL4mult(float newMult){IL4mult = newMult;}
    void setIL8mult(float newMult){IL8mult = newMult;}
    void setIL10mult(float newMult){IL10mult = newMult;}
    void setIL12mult(float newMult){IL12mult = newMult;}
    void setGCSFmult(float newMult){GCSFmult = newMult;}
};



#endif //IIRABM_C_OBJECT_SIMULATION_H
