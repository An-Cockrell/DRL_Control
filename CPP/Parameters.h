	const int xDim=101;
	const int yDim=101;

	const float antibioticMultiplier=-1; // change to 0.2 to activate antibiotics
	// const float antibioticMultiplier=0.2;
	const int parameterInput=0; //number of parameter sets

	const int numTimeSteps=10000; //5760=28 days
	//717 for 3 interventions 1 day apart
	//512 for 2 interventions 1 d apart

	const int numReseeds=0;
//        const int numReseeds=100;
	const int reseedTime=505;

	const int injuryStep=205;

	const int cellCapacity=28;

	const int movie=0; //Dont use with a parameter sweep
	const int movieStep=205;
	const int oxyInfectFinal=0;
	const int dailyOutput=0;
	const int writeEverything=1;

	const string fileName="Outcome_I2_H1_T5_E2_inj33_100seed_int9_99.dat";
	const string fileNameGA="GA_I2_H1_T5_E2_inj33_10rep_5II_length99_longrun.dat";

//// GA Parameters
	const int GA=0; //turn GA on/off
	const int GAseed=1;
	const int reseedPostInt=1;
	const int interventionTimeStart=505; //102=12 hrs after injury
	const int interventionTimeSeperation=100;
	const int tournamentSize=2; //tournament selection parameter
	const int ga_popSize=1000;	//initial population size
	const int maxIter=2000; //maximum number of generations
	const int genomeLength=60;
	const int numIndependentInterventions=5; //should equal genomeLength/12
	const int mutationRate=5; //out of 10
	const int replicateMult=10; //number of times to reseed after 1st intervention
	const int interventionLength=99;

//Intervention Test Parameters

	const int testIntervention=1; //test a specific intervention according to the below parameters
	const int testLength=36; //CHECK CODE TO ENSURE CORRECT INTERVENTION APPLIED
	const int numTests=3;
	const int testStart=205;
	const string intFile="intervention8_truncatedat3.dat";




//Parameters:
//xDim,yDim => The dimensions of the grid in units of endothelial cells;
//	the model has been calibrated for 101x101
//antibioticMultiplier=>infection variable in each cell is multiplied by this factor each
//	time antibiotics are applied; set to <0 for no antibiotic application.
//parameterInput=> number of parameter sets to be read in from a file.  For automatic parameter distribution, set to 0
//numTimeSteps => the number of time steps per simulation
//inj_number => size of initial infectious injury; inj)numberMax is the maximum size of the
//			injury when performing a parameter sweep
//oxyheal => the amount of oxygen healing when 30<oxy<60
//seed => the random number seed
//injuryStep => the amount of steps between recurring small injuries.  Each time step is 7 mins
//infectSpread => measure of how much infection an infected cell spreads to neighboring cells
//numInfectRepeat => how many neighboring cells an infected cell will infect
//cellCapacity => the maximum number of immune cells occupying the same space as an endothelial cell
//movie => set to 1 to write a file storing cell locations and chemical gradients for later analysis in MATLAB
//cytoplot => set to 1 to write a data file of total cytokine levels
//oxyInfectTimeCourse => set to 1 to write data file of total oxy deficit and infection levels at each time step
//oxyInfectFinal => set to 1 to write data file of final oxy deficit and infection values
