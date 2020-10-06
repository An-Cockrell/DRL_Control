#include <vector>
#include <random>
#include <stdlib.h>
#include <algorithm>
#include "agents.h"
#include "Simulation.h"





using namespace std;

extern const int cellCapacity,xDim,yDim,injuryStep,parameterInput,numTimeSteps;
extern const float antibioticMultiplier;
extern void heal(int index,vector<EC>& ecArr);
extern void applyAntibiotics(vector<EC>& ecArr);
extern void adjustOrientation(int* orientation, int leftOrRight);
extern void wiggle(int* orientation);
extern void move(int orient, int* x, int* y, int (&cellGrid)[101][101]);
extern void getAhead(int orient, int x, int y, int *xl, int *xm, int *xr, int *yl, int *ym, int *yr);


void readParams(int procID, int nProc, int p1[], int* np1, float p2[], int* np2, int p3[],
                int* np3, int p4[], int* np4,int p5[], int* np5, int p6[], int* np6);


extern "C" {
    void* simulationObjectAddress;
    SimulationObject sim_obj;
    void * CreateInstance(float OH, int IS, int NRI, int NIR, int InjNum, int inputSeed, int numCyto, float *IP){
        sim_obj = SimulationObject(OH, IS, NRI, NIR, InjNum, inputSeed, numCyto, IP);
        simulationObjectAddress = &sim_obj;
        return simulationObjectAddress;
    }
    void* CopyInstance(void* ptr){
        // SimulationObject *returnSim = new SimulationObject();
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        SimulationObject& tempRef = *ref;
        SimulationObject* returnSim = new SimulationObject(tempRef);
        return returnSim;
    }
    void singleStep(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->singleStep();
    }
    void giveABX(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->giveABX();
    }
    int getSimulationStep(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getStep();
    }
    float getOxydef(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getOxydef();
    }
    float getSystemOxy(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getSystemOxy();
    }
    float getTotalInfection(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotalInfection();
    }
    float getTotal_TNF(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_TNF();
    }
    float getTotal_sTNFr(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_sTNFr();
    }
    float getTotal_IL10(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_IL10();
    }
    float getTotal_GCSF(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_GCSF();
    }
    float getTotal_proTH1(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_proTH1();
    }
    float getTotal_proTH2(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_proTH2();
    }
    float getTotal_IFNg(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_IFNg();
    }
    float getTotal_PAF(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_PAF();
    }
    float getTotal_IL1(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_IL1();
    }
    float getTotal_IL4(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_IL4();
    }
    float getTotal_IL8(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_IL8();
    }
    float getTotal_IL12(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_IL12();
    }
    float getTotal_sIL1r(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_sIL1r();
    }
    float getTotal_IL1ra(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTotal_IL1ra();
    }
    float getPAFmult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getPAFmult();
    }
    float getTNFmult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getTNFmult();
    }
    float getsTNFrmult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getsTNFrmult();
    }
    float getIL1ramult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getIL1ramult();
    }
    float getsIL1rmult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getsIL1rmult();
    }
    float getIFNgmult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getIFNgmult();
    }
    float getIL1mult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getIL1mult();
    }
    float getIL4mult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getIL4mult();
    }
    float getIL8mult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getIL8mult();
    }
    float getIL10mult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getIL10mult();
    }
    float getIL12mult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getIL12mult();
    }
    float getGCSFmult(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getGCSFmult();
    }
    float* getAllSignalsReturn(void* ptr){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        return ref->getAllSignalsReturn();
    }
    void setSeed(void* ptr, int newSeed){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setSeed(newSeed);
    }
    void setPAFmult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setPAFmult(newMult);
    }
    void setTNFmult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setTNFmult(newMult);
    }
    void setsTNFrmult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setsTNFrmult(newMult);
    }
    void setIL1ramult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setIL1ramult(newMult);
    }
    void setsIL1rmult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setsIL1rmult(newMult);
    }
    void setIFNgmult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setIFNgmult(newMult);
    }
    void setIL1mult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setIL1mult(newMult);
    }
    void setIL4mult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setIL4mult(newMult);
    }
    void setIL8mult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setIL8mult(newMult);
    }
    void setIL10mult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setIL10mult(newMult);
    }
    void setIL12mult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setIL12mult(newMult);
    }
    void setGCSFmult(void* ptr, float newMult){
        SimulationObject * ref = reinterpret_cast<SimulationObject *>(ptr);
        ref->setGCSFmult(newMult);
    }

}

SimulationObject::SimulationObject(float OH, int IS, int NRI, int NIR, int InjNum, int inputSeed, int numCyto,
                                   float *IP){
    oxyHeal = OH;
    infectSpread = IS;
    numRecurInj = NRI;
    numInfectRepeat = NIR;
    inj_number = InjNum;
    seed = inputSeed;
    numCytokines = numCyto;
    internalParameterization = IP;

    flag=0;
    generator.seed(seed);

    initialize();
    PAFmult=internalParameterization[0];
    TNFmult=internalParameterization[1];
    IL1mult=internalParameterization[2];
    IFNgmult=internalParameterization[3];
    IL4mult=internalParameterization[4];
    IL8mult=internalParameterization[5];
    IL10mult=internalParameterization[6];
    IL12mult=internalParameterization[7];
    GCSFmult=internalParameterization[8];
    dOut=0;
    step=0;
    istep=0;
    ecIndexes.clear();
    for(i=0;i<xDim*yDim;i++){
        ecIndexes.push_back(i);
    }
    count6hr=0;
    antibiotic1=0;
    antibiotic2=0;
    numABX=0;
    injure_infectionFRD(inj_number);
    updateSystemOxy(istep);
};
SimulationObject::SimulationObject(const SimulationObject& originalSim){
    ecArray = originalSim.ecArray;
    ecIndexes = originalSim.ecIndexes;
    pmnArray = originalSim.pmnArray;
    monoArray = originalSim.monoArray;
    TH0array = originalSim.TH0array;
    TH1array = originalSim.TH1array;
    TH2array = originalSim.TH2array;
    pmn_marrowArray = originalSim.pmn_marrowArray;
    mono_marrowArray = originalSim.mono_marrowArray;
    TH0_germArray = originalSim.TH0_germArray;
    TH1_germArray = originalSim.TH1_germArray;
    TH2_germArray = originalSim.TH2_germArray;
    system_oxy = originalSim.system_oxy;
    oxyDeficit = originalSim.oxyDeficit;
    totalInfection = originalSim.totalInfection;
    total_TNF = originalSim.total_TNF;
    total_sTNFr = originalSim.total_sTNFr;
    total_IL10 = originalSim.total_IL10;
    total_GCSF = originalSim.total_GCSF;
    total_proTH1 = originalSim.total_proTH1;
    total_proTH2 = originalSim.total_proTH2;
    total_IFNg = originalSim.total_IFNg;
    total_PAF = originalSim.total_PAF;
    total_IL1 = originalSim.total_IL1;
    total_IL4 = originalSim.total_IL4;
    total_IL8 = originalSim.total_IL8;
    total_IL12 = originalSim.total_IL12;
    total_sIL1r = originalSim.total_sIL1r;
    total_IL1ra = originalSim.total_IL1ra;
    PAFmult = originalSim.PAFmult;
    TNFmult = originalSim.TNFmult;
    sTNFrmult = originalSim.sTNFrmult;
    IL1ramult = originalSim.IL1ramult;
    sIL1rmult = originalSim.sIL1rmult;
    IFNgmult = originalSim.IFNgmult;
    IL1mult = originalSim.IL1mult;
    IL4mult = originalSim.IL4mult;
    IL8mult = originalSim.IL8mult;
    IL10mult = originalSim.IL10mult;
    IL12mult = originalSim.IL12mult;
    GCSFmult = originalSim.GCSFmult;
    **cellGrid = **originalSim.cellGrid;
    int rows = sizeof(originalSim.cellGrid) / sizeof(originalSim.cellGrid[0]);
    int cols = sizeof(originalSim.cellGrid[0]);
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++){
        cellGrid[r][c] = originalSim.cellGrid[r][c];
      }
    }
    **allSignals = **originalSim.allSignals;
    *allSignalsReturn = *originalSim.allSignalsReturn;
    oxyHeal = originalSim.oxyHeal;
    infectSpread = originalSim.infectSpread;
    numRecurInj = originalSim.numRecurInj;
    numInfectRepeat = originalSim.numInfectRepeat;
    inj_number = originalSim.inj_number;
    seed = originalSim.seed;
    numCytokines = originalSim.numCytokines;
    internalParameterization = originalSim.internalParameterization;
    i = originalSim.i;
    step = originalSim.step;
    iend = originalSim.iend;
    jend = originalSim.jend;
    dOut = originalSim.dOut;
    count6hr = originalSim.count6hr;
    antibiotic1 = originalSim.antibiotic1;
    antibiotic2 = originalSim.antibiotic2;
    istep = originalSim.istep;
    x1 = originalSim.x1;
    x2 = originalSim.x2;
    k = originalSim.k;
    j = originalSim.j;
    numABX = originalSim.numABX;
    flag = originalSim.flag;
}

void SimulationObject::singleStep(){
    step++;
    istep++;
    if(step==injuryStep){step=1;}
    antibiotic1++;
    antibiotic2++;
    updateTrajectoryOutput(istep);
    simulationStep();
    updateSystemOxy(istep);
}
void SimulationObject::simulationStep(){
    cellStep();
    evaporate();
    recurrentInjury();
    diffuse();
    if(antibioticMultiplier>0){
        this->giveABX();
    }
}
void SimulationObject::updateTrajectoryOutput(int q){
    allSignals[0][q]=oxyDeficit;
    allSignals[1][q]=totalInfection;
    allSignals[2][q]=total_TNF;
    allSignals[3][q]=total_sTNFr;
    allSignals[4][q]=total_IL10;
    allSignals[5][q]=total_GCSF;
    allSignals[6][q]=total_proTH1;
    allSignals[7][q]=total_proTH2;
    allSignals[8][q]=pmnArray.size();
    allSignals[9][q]=monoArray.size();
    allSignals[10][q]=TH1array.size();
    allSignals[11][q]=TH2array.size();
    allSignals[12][q]=total_IFNg;
    allSignals[13][q]=total_PAF;
    allSignals[14][q]=total_IL1;
    allSignals[15][q]=total_IL4;
    allSignals[16][q]=total_IL8;
    allSignals[17][q]=total_IL12;
    allSignals[18][q]=total_sIL1r;
    allSignals[19][q]=total_IL1ra;
}


void SimulationObject::updateSystemOxy(int step){
    int size,i;

    system_oxy=0;
    oxyDeficit=0;
    totalInfection=0;
    total_TNF=0;
    total_sTNFr=0;
    total_IL10=0;
    total_GCSF=0;
    total_proTH1=0;
    total_proTH2=0;
    total_IFNg=0;
    total_PAF=0;
    total_IL1=0;
    total_IL4=0;
    total_IL8=0;
    total_IL12=0;
    total_sIL1r=0;
    total_IL1ra=0;
    size=ecArray.size();
    for(i=0;i<size;i++){
        system_oxy+=(ecArray[i].oxy/100);
        totalInfection+=(ecArray[i].infection/100);
        total_TNF+=(ecArray[i].TNF/100);
        total_sTNFr+=(ecArray[i].sTNFr/100);
        total_IL10+=(ecArray[i].IL10/100);
        total_GCSF+=(ecArray[i].GCSF/100);
        total_IFNg+=(ecArray[i].IFNg/100);
        total_PAF+=(ecArray[i].PAF/100);
        total_IL1+=(ecArray[i].IL1/100);
        total_IL4+=(ecArray[i].IL4/100);
        total_IL8+=(ecArray[i].IL8/100);
        total_IL12+=(ecArray[i].IL12/100);
        total_sIL1r+=(ecArray[i].sIL1r/100);
        total_IL1ra+=(ecArray[i].IL1ra/100);
    }
    size=TH0array.size();
    for(i=0;i<size;i++){
        total_proTH1+=(TH0array[i].proTH1/100);
        total_proTH2+=(TH0array[i].proTH2/100);
    }
    oxyDeficit=(xDim*yDim)-system_oxy;
}

void SimulationObject::initialize(){
    int i,j,k,xTemp,yTemp;
//	cout<<"ENTERING INITIALIZE\n";
    ecArray.clear();
    pmnArray.clear();
    monoArray.clear();
    TH0array.clear();
    TH1array.clear();
    TH2array.clear();
    pmn_marrowArray.clear();
    mono_marrowArray.clear();
    TH0_germArray.clear();
    TH1_germArray.clear();
    TH2_germArray.clear();

    system_oxy=10201;
    oxyDeficit=0;
    totalInfection=0;
    total_TNF=0;
    total_sTNFr=0;
    total_IL10=0;
    total_GCSF=0;
    total_proTH1=0;
    total_proTH2=0;

    PAFmult=1;
    TNFmult=1;
    sTNFrmult=1;
    IL1mult=1;
    sIL1rmult=1;
    IL1ramult=1;
    IFNgmult=1;
    IL4mult=1;
    IL8mult=1;
    IL10mult=1;
    IL12mult=1;
    GCSFmult=1;

    for(i=0;i<xDim*yDim;i++){
        ecIndexes.push_back(i);
    }

    for(i=0;i<xDim;i++){
        for(j=0;j<yDim;j++){
            cellGrid[i][j]=0;}}

    k=0; //initialization
    for(j=0;j<yDim;j++){       //Initialize EC grid
        for(i=0;i<xDim;i++){
            ecArray.push_back(EC(i,j,k));
            ecArray[k].getNeighbors();
            k++;
        }
    }

    for(i=0;i<500;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
        pmnArray.push_back(pmn(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }

    for(i=0;i<50;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
        monoArray.push_back(mono(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }

    for(i=0;i<50;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
        TH1array.push_back(TH1(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }

    for(i=0;i<50;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
        TH2array.push_back(TH2(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }

    for(i=0;i<100;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
        pmn_marrowArray.push_back(pmn_marrow(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }

    for(i=0;i<100;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
        mono_marrowArray.push_back(mono_marrow(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }

    for(i=0;i<100;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
        TH0_germArray.push_back(TH0_germ(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }

    for(i=0;i<100;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
        TH1_germArray.push_back(TH1_germ(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }

    for(i=0;i<100;i++){
        xTemp=distribution100(generator);
        yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
        TH2_germArray.push_back(TH2_germ(xTemp,yTemp));
        cellGrid[xTemp][yTemp]++;
    }
}

void SimulationObject::diffuse(){
    float nFactor;
    int i,j,totalCells;

    totalCells=xDim*yDim;
    nFactor=1.0/8.0;

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].endotoxin*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].endotoxin-=ecArray[i].endotoxin;
        ecArray[i].endotoxin+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].PAF*0.6*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].PAF-=(0.6*ecArray[i].PAF);
        ecArray[i].PAF+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].cytotox*0.4*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].cytotox-=(0.4*ecArray[i].cytotox);
        ecArray[i].cytotox+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].TNF*0.6*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].TNF-=(0.6*ecArray[i].TNF);
        ecArray[i].TNF+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].sTNFr*0.8*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].sTNFr-=(0.8*ecArray[i].sTNFr);
        ecArray[i].sTNFr+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL1*0.6*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].IL1-=(0.6*ecArray[i].IL1);
        ecArray[i].IL1+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IFNg*0.8*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].IFNg-=(0.8*ecArray[i].IFNg);
        ecArray[i].IFNg+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL8*0.6*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].IL8-=(0.6*ecArray[i].IL8);
        ecArray[i].IL8+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL10*0.8*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].IL10-=(0.8*ecArray[i].IL10);
        ecArray[i].IL10+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL1ra*0.8*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].IL1ra-=(0.8*ecArray[i].IL1ra);
        ecArray[i].IL1ra+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].sIL1r*0.8*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].sIL1r-=(0.8*ecArray[i].sIL1r);
        ecArray[i].sIL1r+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL12*0.8*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].IL12-=(0.8*ecArray[i].IL12);
        ecArray[i].IL12+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL4*0.8*nFactor);
        }
    }
    for(i=0;i<totalCells;i++){
        ecArray[i].IL4-=(0.8*ecArray[i].IL4);
        ecArray[i].IL4+=ecArray[i].tempSignal;
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].tempSignal=0;
//		cout<<"GCSF="<<ecArray[i].GCSF<<"\n";
    }
    for(i=0;i<totalCells;i++){
        for(j=0;j<8;j++){
            ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].GCSF*nFactor);
        }
    }

    for(i=0;i<totalCells;i++){
        ecArray[i].GCSF-=ecArray[i].GCSF;
        ecArray[i].GCSF+=ecArray[i].tempSignal;
    }
}

void SimulationObject::evaporate(){
    int size,i;
    size=ecArray.size();
    for(i=0;i<size;i++){
        ecArray[i].endotoxin*=0.7;
        if(ecArray[i].endotoxin<0.01){ecArray[i].endotoxin=0;}
        ecArray[i].PAF*=0.7;
        if(ecArray[i].PAF<0.01){ecArray[i].PAF=0;}
        ecArray[i].cytotox*=0.7;
        if(ecArray[i].cytotox<0.01){ecArray[i].cytotox=0;}
        ecArray[i].TNF*=0.8;
        if(ecArray[i].TNF<0.01){ecArray[i].TNF=0;}
        ecArray[i].IL1*=0.8;
        if(ecArray[i].IL1<0.01){ecArray[i].IL1=0;}
        ecArray[i].sTNFr*=0.9;
        if(ecArray[i].sTNFr<0.01){ecArray[i].sTNFr=0;}
        ecArray[i].IL1ra*=0.9;
        if(ecArray[i].IL1ra<0.01){ecArray[i].IL1ra=0;}
        ecArray[i].sIL1r*=0.9;
        if(ecArray[i].sIL1r<0.01){ecArray[i].sIL1r=0;}
        ecArray[i].IFNg*=0.8;
        if(ecArray[i].IFNg<0.01){ecArray[i].IFNg=0;}
        ecArray[i].IL8*=0.7;
        if(ecArray[i].IL8<0.01){ecArray[i].IL8=0;}
        ecArray[i].IL10*=0.95;
        if(ecArray[i].IL10<0.01){ecArray[i].IL10=0;}
        ecArray[i].IL12*=0.8;
        if(ecArray[i].IL12<0.01){ecArray[i].IL12=0;}
        ecArray[i].IL4*=0.95;
        if(ecArray[i].IL4<0.01){ecArray[i].IL4=0;}
        ecArray[i].GCSF*=0.95;
        if(ecArray[i].GCSF<0.01){ecArray[i].GCSF=0;}
    }
}

void SimulationObject::injure_infectionFRD(int inj_number){ //Fixed Radius Disk
    int i,x,y,rad,size,id;
    size=ecArray.size();
    rad=inj_number;
    for(i=0;i<size;i++){
        x=ecArray[i].xLoc-xDim/2;
        y=ecArray[i].yLoc-yDim/2;
        if((pow(x,2)+pow(y,2))<=(pow(rad,2))){
            id=ecArray[i].yLoc*xDim+ecArray[i].xLoc;
            ecArray[id].infection=100;
        }
    }
}

void SimulationObject::recur_injury(){
    int x,y,id;
    x=distribution100(generator);
    y=distribution100(generator);
    id=y*xDim+x;
    ecArray[id].infection=100;
}
void SimulationObject::recurrentInjury(){
    int i;
    if(step==injuryStep-1){
        for(i=1;i<=numRecurInj;i++){
            recur_injury();
        }
    }
}

void SimulationObject::clearIntervention(){
    PAFmult=1;
    TNFmult=1;
    sTNFrmult=1;
    IL1mult=1;
    sIL1rmult=1;
    IL1ramult=1;
    IFNgmult=1;
    IL4mult=1;
    IL8mult=1;
    IL10mult=1;
    IL12mult=1;
    GCSFmult=1;
}

void SimulationObject::giveABX(){
    applyAntibiotics(ecArray);
    numABX++;
}

void SimulationObject::cellStep(){
    int length,j;
    length=TH0array.size();
    if(length>0){
        shuffle(TH0array.begin(),TH0array.end(),generator);}
    j=0;
    while(j<length){
        TH0array[j].TH0function(j, cellGrid, ecArray, TH0array, TH1array, TH2array);
        j++;
        length=TH0array.size();}
    length=ecArray.size();
    shuffle(ecIndexes.begin(),ecIndexes.end(),generator);
    j=0;
    while(j<length){
        ecArray[ecIndexes[j]].inj_function(infectSpread,numInfectRepeat, ecArray);
        ecArray[ecIndexes[j]].ECfunction(oxyHeal, PAFmult, IL8mult, ecArray);
        j++;
        length=ecArray.size();}
    length=pmnArray.size();
    if(length>0){
        shuffle(pmnArray.begin(),pmnArray.end(),generator);}
    j=0;
    while(j<length){
        pmnArray[j].pmn_function(j, TNFmult, IL1ramult, IL1mult, cellGrid, ecArray, pmnArray);
        j++;
        length=pmnArray.size();}
    length=monoArray.size();
    if(length>0){
        shuffle(monoArray.begin(),monoArray.end(),generator);}
    j=0;
    while(j<length){
        monoArray[j].mono_function(j, IL1ramult, sTNFrmult,sIL1rmult, GCSFmult, IL8mult, IL12mult, IL10mult, IL1mult,
                                   TNFmult, cellGrid, ecArray, monoArray);
        j++;
        length=monoArray.size();}
    length=TH1array.size();
    if(length>0){
        shuffle(TH1array.begin(),TH1array.end(),generator);}
    j=0;
    while(j<length){
        TH1array[j].TH1function(j, IFNgmult, cellGrid, ecArray, TH1array);
        j++;
        length=TH1array.size();}
    length=TH2array.size();
    if(length>0){
        shuffle(TH2array.begin(),TH2array.end(),generator);}
    j=0;
    while(j<length){
        TH2array[j].TH2function(j,IL4mult, IL10mult, cellGrid, ecArray, TH2array);
        j++;
        length=TH2array.size();}
    length=pmn_marrowArray.size();
    if(length>0){
        shuffle(pmn_marrowArray.begin(),pmn_marrowArray.end(),generator);}
    j=0;
    while(j<length){
        pmn_marrowArray[j].pmn_marrow_function(total_GCSF, cellGrid, pmnArray);
        j++;
        length=pmn_marrowArray.size();}

    length=mono_marrowArray.size();
    if(length>0){
        shuffle(mono_marrowArray.begin(),mono_marrowArray.end(),generator);}
    j=0;
    while(j<length){
        mono_marrowArray[j].mono_marrow_function(cellGrid, monoArray);
        j++;
        length=mono_marrowArray.size();}

    length=TH1_germArray.size();
    if(length>0){
        shuffle(TH1_germArray.begin(),TH1_germArray.end(),generator);}
    j=0;
    while(j<length){
        TH1_germArray[j].TH1_germ_function(cellGrid, ecArray, TH1array);
        j++;
        length=TH1_germArray.size();}

    length=TH2_germArray.size();
    if(length>0){
        shuffle(TH2_germArray.begin(),TH2_germArray.end(),generator);}
    j=0;
    while(j<length){
        TH2_germArray[j].TH2_germ_function(cellGrid, ecArray, TH2array);
        j++;
        length=TH2_germArray.size();}

    length=TH0_germArray.size();
    if(length>0){
        shuffle(TH0_germArray.begin(),TH0_germArray.end(),generator);}
    j=0;
    while(j<length){
        TH0_germArray[j].TH0_germ_function(cellGrid, ecArray, TH0array, TH1array, TH2array);
        j++;
        length=TH0_germArray.size();
    }
}
float* SimulationObject::getAllSignalsReturn(){
  int c=0;
  // cout << "getting all signals";
  // cout << sizeof(allSignals) / sizeof(allSignals[0]);
  // cout << (sizeof(allSignals[0]) / sizeof(allSignals[0][0]));
  for(int a=0;a<(sizeof(allSignals) / sizeof(allSignals[0])); a++){
    for(int b=0;b<(sizeof(allSignals[0]) / sizeof(allSignals[0][0]));b++){
      allSignalsReturn[c]=allSignals[a][b];
      c++;
    }
  }
  return allSignalsReturn;
}
