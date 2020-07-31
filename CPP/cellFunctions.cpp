#include <stdlib.h>
#include <vector>
#include <iostream>
#include "agents.h"
#include <algorithm>
#include <cmath>
#include <random>
#include "Parameters.h"
//#include "Simulation.h"

using namespace std;


extern mt19937 generator;
extern uniform_int_distribution<int> distribution10k;
extern uniform_int_distribution<int> distribution1000;
extern uniform_int_distribution<int> distribution100;
extern uniform_int_distribution<int> distribution50;
extern uniform_int_distribution<int> distribution12;
extern uniform_int_distribution<int> distribution10;
extern uniform_int_distribution<int> distribution9;
extern uniform_int_distribution<int> distribution8;
extern uniform_int_distribution<int> distribution5;
extern uniform_int_distribution<int> distribution3;
extern uniform_int_distribution<int> distribution2;

void heal(int index,vector<EC>& ecArr);
void applyAntibiotics(vector<EC>& ecArr);
void adjustOrientation(int* orientation, int leftOrRight);
void wiggle(int* orientation);
void move(int orient, int* x, int* y, int (&cellGrid)[101][101]);
void getAhead(int orient, int x, int y, int *xl, int *xm, int *xr, int *yl, int *ym, int *yr);

void EC::inj_function(int infectSpread, int numInfectRepeat, vector<EC>& ecArray){
	int temp,i;

	oxy=max(float(0),oxy-infection);
	endotoxin+=(infection/10);

	for(i=1;i<=numInfectRepeat;i++){
		if(infection>=100){
			temp=distribution8(generator);
			ecArray[neighbors[temp]].infection+=infectSpread;
			infection=100;
	}}
	if(infection>0){
		infection=max(float(0),infection-cytotox+float(0.1));}
}

void EC::activate(float PAFmult, float IL8mult){
	ec_roll++;
	ec_stick++;
	if(PAFmult<=1){
		PAF=(PAF+1)*PAFmult;
	} else{
		PAF=PAF+1+PAFmult-1;
	}
	if(IL8mult<=1){
		IL8=(IL8+1)*IL8mult;
	} else{
		IL8=IL8+1+IL8mult-1;
	}
}

void EC::ECfunction(float oxyHeal, float PAFmult, float IL8mult, vector<EC>& ecArray){
	if((endotoxin>=1)||(oxy<60)){
		ec_activation=1;}
	if(ec_activation==1){
		activate(PAFmult, IL8mult);}
	patch_inj_spread(oxyHeal, PAFmult, ecArray);
}

void EC::patch_inj_spread(float oxyHeal, float PAFmult, vector<EC>& ecArray){
	int i;
	oxy=oxy-cytotox;

	if(oxy>=60){
		oxy=min(float(100),oxy+oxyHeal);}

	if((oxy<60)&&(oxy>30)){ //ischemia
		ec_roll++;
		oxy-=0.5;
		if(PAFmult<=1){
			PAF=(PAF+1)*PAFmult;
		} else{
			PAF=PAF+1+PAFmult-1;
		}
		for(i=0;i<8;i++){
			ecArray[neighbors[i]].oxy-=0.05;}
	}
		if(oxy<=30){ //infarction
			ec_stick++;
			oxy-=2;
			if(PAFmult<=1){
				PAF=(PAF+1)*PAFmult;
			} else{
				PAF=PAF+1+PAFmult-1;
			}
		for(i=0;i<8;i++){
			ecArray[neighbors[i]].oxy-=0.25;
			if(ecArray[neighbors[i]].oxy<0){ecArray[neighbors[i]].oxy=0;}
		}
	}
	if(oxy<0){oxy=0;}
}
void EC::getNeighbors(){
	int nw,n,ne,e,se,s,sw,w;

	nw=id+xDim-1;
	n=id+xDim;
	ne=id+xDim+1;
	e=id+1;
	se=id-xDim+1;
	s=id-xDim;
	sw=id-xDim-1;
	w=id-1;

	if(id%xDim==0){ //object lies on west grid border
		nw=id+2*xDim-1;
		w=id+xDim-1;
		sw=id-1;
	}

	if(id%xDim==xDim-1){ //object lies on east grid border
		ne=id+1;
		e=id-(xDim-1);
		se=id-2*xDim+1;
	}

	if(id+xDim>=xDim*yDim){ //object lies on northern border
		n=(id+xDim)%(xDim*yDim);
		ne=(id+xDim)%(xDim*yDim)+1;
		nw=(id+xDim)%(xDim*yDim)-1;
		if(id%xDim==0){
			nw=(id+xDim)%(xDim*yDim)+xDim-1;
		}
		if(id%xDim==xDim-1){
			ne=(id+xDim)%(xDim*yDim)-(xDim-1);
		}
	}

	if(id<xDim){ //object lies on southern border
		s=(xDim*yDim)-xDim+id;
		se=(xDim*yDim)-xDim+id+1;
		sw=(xDim*yDim)-xDim+id-1;
		if(id%xDim==0){
			sw=(xDim*yDim)-1;
		}
		if(id%xDim==xDim-1){
			se=(xDim*yDim)-xDim;
		}
	}

	neighbors[0]=nw;
	neighbors[1]=n;
	neighbors[2]=ne;
	neighbors[3]=e;
	neighbors[4]=se;
	neighbors[5]=s;
	neighbors[6]=sw;
	neighbors[7]=w;
}


void pmn::pmn_function(int pmnID, float IL1ramult, float TNFmult, float IL1mult, int (&cellGrid)[101][101], vector<EC>& ecArray,
		vector<pmn>& pmnArray){
	int roll,x,y,id;
	float TNF,PAF,IL1,IL10,GCSF;
	x=xLoc;
	y=yLoc;
	id=y*xDim+x;
	roll=ecArray[id].ec_roll;
	TNF=ecArray[id].TNF;
	PAF=ecArray[id].PAF;
	IL1=ecArray[id].IL1;
	IL10=ecArray[id].IL10;
	GCSF=ecArray[id].GCSF;
	if(wbc_migrate>0){
		pmn_burst(pmnID, TNFmult, IL1mult, cellGrid, ecArray, pmnArray);}
	else{
		if((roll>3)&&(wbc_roll==1)){
			pmn_sniff(cellGrid, ecArray);}
		else{
			pmn_sniff(cellGrid, ecArray);
			pmn_sniff(cellGrid, ecArray);}
		x=xLoc;
		y=yLoc;
		id=y*xDim+x;
		TNF=ecArray[id].TNF;
		PAF=ecArray[id].PAF;
		IL1=ecArray[id].IL1;
		IL10=ecArray[id].IL10;
		GCSF=ecArray[id].GCSF;
		if(TNF+PAF>1){
			wbc_stick=IL1;
			if(IL1ramult<=1){
				ecArray[id].IL1ra=(ecArray[id].IL1ra+1)*IL1ramult;
			} else{
				ecArray[id].IL1ra=ecArray[id].IL1ra+1+IL1ramult-1;
			}
		}
		if((wbc_stick>=1)&&(ecArray[id].ec_stick>=100)){
			wbc_migrate=max(float(0),(TNF+IL1+GCSF-IL10));}
		pmn_age--;
		if(pmn_age<0){
			x=xLoc;
			y=yLoc;
			pmnArray.erase(pmnArray.begin()+pmnID);
			cellGrid[x][y]--;
			if(cellGrid[x][y]<0){cout<<"Cell Error PMNFUNC\n";
			return;
			}}
	}
}

void pmn::pmn_burst(int pmnID, float TNFmult, float IL1mult, int (&cellGrid)[101][101], vector<EC>& ecArray,
		vector<pmn>& pmnArray){
	int x,y,id;

	x=xLoc;
	y=yLoc;
	id=y*xDim+x;
	ecArray[id].cytotox=max(float(10),ecArray[id].TNF);
	ecArray[id].oxy=100;
	ecArray[id].ec_roll=0;
	ecArray[id].ec_stick=0;
	ecArray[id].ec_migrate=0;
	if(TNFmult<=1){
		ecArray[id].TNF=(ecArray[id].TNF+1)*TNFmult;
	} else{
		ecArray[id].TNF=ecArray[id].TNF+1+TNFmult-1;
	}
	if(IL1mult<=1){
		ecArray[id].IL1=(ecArray[id].IL1+1)*IL1mult;
	} else{
		ecArray[id].IL1=ecArray[id].IL1+1+IL1mult-1;
	}
	pmn_age=pmn_pcd;
	pmn_pcd=pmn_pcd-1+max(float(0),(ecArray[id].TNF+ecArray[id].IFNg+ecArray[id].GCSF-
		ecArray[id].IL10)/100);
	if(pmn_age<0){
		x=xLoc;
		y=yLoc;
		pmnArray.erase(pmnArray.begin()+pmnID);
		cellGrid[x][y]--;
	}
}

void pmn::pmn_sniff(int (&cellGrid)[101][101], vector<EC>& ecArray){
	int x,y,idm,idr,idl,xl,xm,xr,yl,ym,yr,flag;
	float pmnahead,pmnright,pmnleft;
	x=xLoc;
	y=yLoc;

	flag=0;  //Flag=-1 for left, 0 for middle, and 1 for right

	getAhead(orientation, x, y, &xl, &xm, &xr, &yl, &ym, &yr);

	idr=yr*xDim+xr;
	idm=ym*xDim+xm;
	idl=yl*xDim+xl;
	pmnright=ecArray[idr].IL8;
	pmnahead=ecArray[idm].IL8;
	pmnleft=ecArray[idl].IL8;

	if((pmnright>=pmnahead)&&(pmnright>=pmnleft)){flag=1;}
	else if(pmnleft>=pmnahead){flag-=1;}
	adjustOrientation(&orientation,flag);

	if(flag==-1){
		if(cellGrid[xl][yl]<cellCapacity){
 			xLoc=xl;
 			yLoc=yl;
 			cellGrid[xl][yl]++;
 			cellGrid[x][y]--;}
	}
	if(flag==0){
		if(cellGrid[xm][ym]<cellCapacity){
 			xLoc=xm;
 			yLoc=ym;
 			cellGrid[xm][ym]++;
 			cellGrid[x][y]--;}
	}
	if(flag==1){
		if(cellGrid[xr][yr]<cellCapacity){
 			xLoc=xr;
 			yLoc=yr;
 			cellGrid[xr][yr]++;
 			cellGrid[x][y]--;}
	}
}
void pmn_marrow::pmn_marrow_function(float total_GCSF, int (&cellGrid)[101][101], vector<pmn>& pmnArray){
	int x,y,temp,n,i;
	if(total_GCSF>3000){total_GCSF=3000;}
	n=int(1+total_GCSF/100);

	for(i=0;i<n;i++){
		x=distribution100(generator);
		y=distribution100(generator);
		temp=distribution10(generator);
		if(temp<1){
			pmnArray.push_back(pmn(x,y,50));
			cellGrid[x][y]++;
		}
	}
}

void mono::mono_function(int index, float IL1ramult, float sTNFrmult, float sIL1rmult, float GCSFmult, float
IL8mult, float IL12mult, float IL10mult, float IL1mult, float TNFmult, int (&cellGrid)[101][101], vector<EC>&
        ecArray, vector<mono>& monoArray){
	int x,y,id;

	x=xLoc;
	y=yLoc;
	id=y*xDim+x;

	if(ecArray[id].sTNFr<=100){
		TNFr=min(float(100),(ecArray[id].TNF+ecArray[id].sTNFr));}
	else{
		TNFr=min(float(100),max(float(0),ecArray[id].TNF-ecArray[id].sTNFr));}
	IL_1r=min(float(100),max(float(0),ecArray[id].IL1-ecArray[id].IL1ra-ecArray[id].sIL1r));

	if(IL1ramult<=1){
		ecArray[id].IL1ra=(ecArray[id].IL1ra+ecArray[id].IL1/2)*IL1ramult;
	} else {
		ecArray[id].IL1ra=(ecArray[id].IL1ra+ecArray[id].IL1/2)+IL1ramult-1;
	}
	if(sTNFrmult<=1){
		ecArray[id].sTNFr=(ecArray[id].sTNFr+TNFr/2)*sTNFrmult;
	} else {
		ecArray[id].sTNFr=(ecArray[id].sTNFr+TNFr/2)+sTNFrmult-1;
	}
	if(sIL1rmult<=1){
		ecArray[id].sIL1r=(ecArray[id].sIL1r+IL_1r/2)*sIL1rmult;
	} else {
		ecArray[id].sIL1r=(ecArray[id].sIL1r+IL_1r/2)+sIL1rmult-1;
	}
	activation=ecArray[id].endotoxin+ecArray[id].PAF+ecArray[id].IFNg-ecArray[id].IL10;
	if(activation>0){
		if(GCSFmult<=1){
			ecArray[id].GCSF=(ecArray[id].GCSF+ecArray[id].endotoxin+ecArray[id].PAF+ecArray[id].TNF+
			ecArray[id].IFNg)*GCSFmult;
		} else {
			ecArray[id].GCSF=(ecArray[id].GCSF+ecArray[id].endotoxin+ecArray[id].PAF+ecArray[id].TNF+
			ecArray[id].IFNg)+GCSFmult-1;
		}
		if(IL8mult<=1){
			ecArray[id].IL8=(ecArray[id].IL8+ecArray[id].TNF+ecArray[id].IL1)*IL8mult;
		} else {
			ecArray[id].IL8=(ecArray[id].IL8+ecArray[id].TNF+ecArray[id].IL1)+IL8mult-1;
		}
		if(IL12mult<=1){
			ecArray[id].IL12=(ecArray[id].IL12+ecArray[id].TNF+ecArray[id].IL1)*IL12mult;
		} else {
			ecArray[id].IL12=(ecArray[id].IL12+ecArray[id].TNF+ecArray[id].IL1)+IL12mult;
		}
		if(IL10mult<=1){
			ecArray[id].IL10=(ecArray[id].IL10+ecArray[id].TNF+ecArray[id].IL1)*IL10mult;
		} else {
			ecArray[id].IL10=(ecArray[id].IL10+ecArray[id].TNF+ecArray[id].IL1)+IL10mult-1;
		}
		if(IL1mult<=1){
			ecArray[id].IL1=(ecArray[id].IL1+ecArray[id].endotoxin+ecArray[id].PAF+IL_1r+
			ecArray[id].TNF)*IL1mult;
		} else {
			ecArray[id].IL1=(ecArray[id].IL1+ecArray[id].endotoxin+ecArray[id].PAF+IL_1r+
			ecArray[id].TNF)+IL1mult-1;
		}
		if(TNFmult<=1){
			ecArray[id].TNF=(ecArray[id].TNF+ecArray[id].endotoxin+ecArray[id].PAF+TNFr+
			ecArray[id].IFNg)*TNFmult;
		} else {
			ecArray[id].TNF=(ecArray[id].TNF+ecArray[id].endotoxin+ecArray[id].PAF+TNFr+
			ecArray[id].IFNg)+TNFmult-1;
		}
		if((wbc_stick==1)&&(ecArray[id].ec_stick>=100)){
			wbc_migrate=1;}
		if(wbc_roll==1){
			wbc_stick=1;}
		wbc_roll=1;
	}
	if(activation<0){
		if(IL10mult<=1){
			ecArray[id].IL10=(ecArray[id].IL10+ecArray[id].TNF+ecArray[id].IL1)*IL10mult;
		} else {
			ecArray[id].IL10=(ecArray[id].IL10+ecArray[id].TNF+ecArray[id].IL1)+IL10mult-1;
		}
	}
	if(wbc_migrate==1){
		heal(id, ecArray);}
	if(wbc_roll==1){
		mono_sniff(cellGrid, ecArray);}
	else{
		mono_sniff(cellGrid, ecArray);
		mono_sniff(cellGrid, ecArray);}
	mono_age--;
	if(mono_age<0){
		x=xLoc;
		y=yLoc;
		monoArray.erase(monoArray.begin()+index);
		cellGrid[x][y]--;
	}
	if(activation>20){
		activation=20;}
}

void mono::mono_sniff(int (&cellGrid)[101][101], vector<EC>& ecArray){
	int x,y,idm,idr,idl,xl,xm,xr,yl,ym,yr,flag;
	float pafahead,pafright,pafleft;
	x=xLoc;
	y=yLoc;

	flag=0;
	getAhead(orientation, x, y, &xl, &xm, &xr, &yl, &ym, &yr);
	idr=yr*xDim+xr;
	idm=ym*xDim+xm;
	idl=yl*xDim+xl;
	pafright=ecArray[idr].PAF;
	pafahead=ecArray[idm].PAF;
	pafleft=ecArray[idl].PAF;

	if((pafright>=pafahead)&&(pafright>=pafleft)){flag=1;}
	else if(pafleft>=pafahead){flag-=1;}
	adjustOrientation(&orientation,flag);

	if(flag==-1){
		if(cellGrid[xl][yl]<cellCapacity){
 			xLoc=xl;
 			yLoc=yl;
 			cellGrid[xl][yl]++;
 			cellGrid[x][y]--;}
	}

	if(flag==0){
		if(cellGrid[xm][ym]<cellCapacity){
 			xLoc=xm;
 			yLoc=ym;
 			cellGrid[xm][ym]++;
 			cellGrid[x][y]--;}
	}

	if(flag==1){
		if(cellGrid[xr][yr]<cellCapacity){
 			xLoc=xr;
 			yLoc=yr;
 			cellGrid[xr][yr]++;
 			cellGrid[x][y]--;}
	}
}

void mono_marrow::mono_marrow_function(int (&cellGrid)[101][101], vector<mono>& monoArray){
	int x,y,temp;
	x=distribution100(generator);
	y=distribution100(generator);
	temp=distribution100(generator);
	if(temp<1){
		monoArray.push_back(mono(x,y,50,1000,1000));
		cellGrid[x][y]++;
	}
}


void TH0::TH0function(int index, int (&cellGrid)[101][101], vector<EC>& ecArray, vector<TH0>& TH0array, vector<TH1>&
        TH1array, vector<TH2>& TH2array){
	int id;
//	cout<<"activation="<<activation<<"\n";
	id=yLoc*xDim+xLoc;
	if(ecArray[id].IL12+ecArray[id].IL4>0){
		proTH1=(ecArray[id].IL12+ecArray[id].IFNg)*100;
		proTH2=(ecArray[id].IL10+ecArray[id].IL4)*100;
		if((proTH1>0)&&(proTH2>0)){
			rTH1=distribution10k(generator)%int(ceil(proTH1));
			rTH2=distribution10k(generator)%int(ceil(proTH2));
			if(rTH1>rTH2){activation++;}
			if(rTH1<rTH2){activation--;}
		}
		if(proTH1==0){activation--;}
		if(proTH2==0){activation++;}
	}
	wiggle(&orientation);
	move(orientation, &xLoc, &yLoc, cellGrid);
	TH0_age--;
	if(TH0_age<0){
		cellGrid[xLoc][yLoc]--;
		TH0array.erase(TH0array.begin()+index);
		return;}
	if(activation>=10){
		TH1array.push_back(TH1(xLoc,yLoc,TH0_age));
		TH0array.erase(TH0array.begin()+index);
	}
	if(activation<=-10){
		TH2array.push_back(TH2(xLoc,yLoc,TH0_age));
		TH0array.erase(TH0array.begin()+index);
	}
}

void TH1::TH1function(int index, float IFNgmult, int (&cellGrid)[101][101],vector<EC>& ecArray, vector<TH1>& TH1array){
	int id;

	id=yLoc*xDim+xLoc;
	if(ecArray[id].IL12>0){
		if(IFNgmult<=1){
			ecArray[id].IFNg=(2*ecArray[id].IFNg+ecArray[id].IL12+ecArray[id].TNF+
			ecArray[id].IL1)*IFNgmult;
		} else {
			ecArray[id].IFNg=(2*ecArray[id].IFNg+ecArray[id].IL12+ecArray[id].TNF+
			ecArray[id].IL1)+IFNgmult-1;
		}
	}
	wiggle(&orientation);
	move(orientation, &xLoc, &yLoc, cellGrid);
	TH1_age--;
	if(TH1_age<0){
		cellGrid[xLoc][yLoc]--;
		TH1array.erase(TH1array.begin()+index);
	}
}

void TH2::TH2function(int index, float IL4mult, float IL10mult, int (&cellGrid)[101][101], vector<EC>& ecArray,
		vector<TH2>& TH2array){
	int id;

	id=yLoc*xDim+xLoc;

	if(ecArray[id].IL10>0){
		if(IL4mult<=1){
			ecArray[id].IL4=(ecArray[id].IL4+ecArray[id].IL10)*IL4mult;
		} else {
			ecArray[id].IL4=(ecArray[id].IL4+ecArray[id].IL10)+IL4mult-1;
		}
		if(IL10mult<=1){
			ecArray[id].IL10=ecArray[id].IL10*2*IL10mult;
		} else {
			ecArray[id].IL10=ecArray[id].IL10*2+IL10mult-1;
		}
	}
	wiggle(&orientation);
	move(orientation, &xLoc, &yLoc, cellGrid);
	TH2_age--;
	if(TH2_age<0){
		cellGrid[xLoc][yLoc]--;
		TH2array.erase(TH2array.begin()+index);

	}
}

void TH0_germ::TH0_germ_function(int (&cellGrid)[101][101], vector<EC>& ecArray, vector<TH0>& TH0array, vector<TH1>& TH1array, vector<TH2>& TH2array){
	int x,y,temp;
	x=xLoc;
	y=yLoc;
	temp=distribution100(generator);
	if(temp<1){
		TH0array.push_back(TH0(x,y,100));
		cellGrid[x][y]++;
	}
}

void TH1_germ::TH1_germ_function(int (&cellGrid)[101][101], vector<EC>& ecArray, vector<TH1>& TH1array){
	int x,y,temp;
	x=xLoc;
	y=yLoc;
	temp=distribution100(generator);
	if(temp<1){
		TH1array.push_back(TH1(x,y,100));
		cellGrid[x][y]++;
	}
}

void TH2_germ::TH2_germ_function(int (&cellGrid)[101][101], vector<EC>& ecArray, vector<TH2>& TH2array){
	int x,y,temp;
	x=xLoc;
	y=yLoc;
	temp=distribution100(generator);
	if(temp<1){
		TH2array.push_back(TH2(x,y,100));
		cellGrid[x][y]++;
	}
}

void heal(int index, vector<EC>& ecArr){
	ecArr[index].oxy=100;
	ecArr[index].ec_roll=0;
	ecArr[index].ec_stick=0;
	ecArr[index].ec_migrate=0;
	ecArr[index].infection=0;
	ecArr[index].ec_activation=0;
}

void applyAntibiotics(vector<EC>& ecArr){
	int i,length;
	length=ecArr.size();

	for(i=0;i<length;i++){
		ecArr[i].infection=ecArr[i].infection*antibioticMultiplier;
	}
}

void adjustOrientation(int* orientation, int leftOrRight){
//if leftOrRight=-1, adjust orientation left, if =1, adjust orientation right
    int tempOrient;
    tempOrient=*orientation;
    tempOrient+=leftOrRight;
    if(tempOrient>7){tempOrient=0;}
    if(tempOrient<0){tempOrient=7;}
    *orientation=tempOrient;
}

void wiggle(int* orientation){ //Should always be followed by move() to match NetLogo
    int dir,tempOrient;
    tempOrient=*orientation;
    dir=distribution3(generator);
    if(dir==0){tempOrient--;}
    if(dir==2){tempOrient++;}
    if(tempOrient>7){tempOrient=0;}
    if(tempOrient<0){tempOrient=7;}
    *orientation=tempOrient;
}

void move(int orient, int* x, int* y, int (&cellGrid)[101][101]){
	int oldx,newx,oldy,newy;

	oldx=*x;
	oldy=*y;
	newx=oldx;
	newy=oldy;

	if(orient==0){ //Move North
		newy=oldy+1;
		if(newy>=yDim){newy=0;}
	}

	if(orient==1){ //Move Northeast
		newy=oldy+1;
		newx=oldx+1;
		if(newy>=yDim){newy=0;}
		if(newx>=yDim){newx=0;}
	}

	if(orient==2){ //Move East
		newx=oldx+1;
		if(newx>=xDim){newx=0;}
	}

	if(orient==3){ //Move Southeast
		newy=oldy-1;
		newx=oldx+1;
		if(newx>=xDim){newx=0;}
		if(newy<0){newy=yDim-1;}
	}

	if(orient==4){ //Move South
		newy=oldy-1;
		if(newy<0){newy=yDim-1;}
	}

	if(orient==5){ //Move Southwest
		newy=oldy-1;
		newx=oldx-1;
		if(newy<0){newy=yDim-1;}
		if(newx<0){newx=xDim-1;}
	}

	if(orient==6){ //Move West
		newx=oldx-1;
		if(newx<0){newx=xDim-1;}
	}

	if(orient==7){ //Move Northwest
		newx=oldx-1;
		newy=oldy+1;
		if(newx<0){newx=xDim-1;}
		if(newy>=yDim){newy=0;}
	}

	if(cellGrid[newx][newy]<cellCapacity){
		*x=newx;
		*y=newy;
		cellGrid[oldx][oldy]--;
		cellGrid[newx][newy]++;
	}
}

void getAhead(int orient, int x, int y, int *xl, int *xm, int *xr, int *yl, int *ym, int *yr){
	int txl,txm,txr,tyl,tym,tyr;

	if(orient==0){
		txl=x-1;
		txm=x;
		txr=x+1;
		tyl=y+1;
		tym=y+1;
		tyr=y+1;
	}

	if(orient==1){
		txl=x;
		txm=x+1;
		txr=x+1;
		tyl=y+1;
		tym=y+1;
		tyr=y;
	}

	if(orient==2){
		txl=x+1;
		txm=x+1;
		txr=x+1;
		tyl=y+1;
		tym=y;
		tyr=y-1;
	}

	if(orient==3){
		txl=x+1;
		txm=x+1;
		txr=x;
		tyl=y;
		tym=y-1;
		tyr=y-1;
	}

	if(orient==4){
		txl=x+1;
		txm=x;
		txr=x-1;
		tyl=y-1;
		tym=y-1;
		tyr=y-1;
	}

	if(orient==5){
		txl=x;
		txm=x-1;
		txr=x-1;
		tyl=y-1;
		tym=y-1;
		tyr=y;
	}

	if(orient==6){
		txl=x-1;
		txm=x-1;
		txr=x-1;
		tyl=y-1;
		tym=y;
		tyr=y+1;
	}

	if(orient==7){
		txl=x-1;
		txm=x-1;
		txr=x;
		tyl=y;
		tym=y+1;
		tyr=y+1;
	}

	if(txl<0){txl=xDim-1;}
	if(txl>=xDim){txl=0;}
	if(txm<0){txm=xDim-1;}
	if(txm>=xDim){txm=0;}
	if(txr<0){txr=xDim-1;}
	if(txr>=xDim){txr=0;}
	if(tyl<0){tyl=yDim-1;}
	if(tyl>=yDim){tyl=0;}
	if(tym<0){tym=yDim-1;}
	if(tym>=yDim){tym=0;}
	if(tyr<0){tyr=yDim-1;}
	if(tyr>=yDim){tyr=0;}

	*xl=txl;
	*xm=txm;
	*xr=txr;
	*yl=tyl;
	*ym=tym;
	*yr=tyr;
}