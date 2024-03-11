#Version 1.2

import imaginaryEigen;
import numpy as np;
import matplotlib.pyplot as plt;

plt.rcParams.update({'font.size': 16})





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Simulation Parameters Block   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Simulation Defining Parameters
simulationName = "testingFunc";
splittingType = "CPR86SS";
integrationScheme = "LS7";
numStates = 10;
bCalculateEnergy = True;
bMomentumStates = False;
bAllowCaching = False;
bWriteEigenfunctions = False;
bWriteEnergies = False;

#System Defining Parameters
spacialNumber = 1800;
bounds = [-60,60];
dx = (bounds[1] - bounds[0])/(spacialNumber);
dt = 0.01;
finalTime = 100.0000;
order = 2.0;

#Adaptive Stepping Parameters
bAdaptiveSteping = True;
steppingProduct = 1;

#Symmetry Breaking Strategies
bParityBreaking = False;
bEven = True;

#Accuracy Parameters
cauchyConvergenceTolerence = 10e-14;
edgeTolerance = 10e-6;                  #Depreciated

#Timer Staging Parameters
bEvolutionStaging = False;
timeDescritizations = [10e-4];
times = [10e-1];

#Simulation Plotting Parameters
bPlot = False;
bCreateGif = False;                     #Depreciated
bCreatePhaseGif = False;                #Depreciated
frameRate = 15;                         #Depreciated
secondsOfGif = 12;                      #Depreciated

#User Feedback Parameters
bUserOutput = True;
bTimingSingle = True;
bTimingTotal = True;


#Reset Parameter 
bReset = False;     #(KEEP FALSE IF YOU DON'T WANT TO REMOVE DATA)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   DEFINING INITAL COND. AND POTENTIAL   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Define spacial grid for use in defining potentials and initial conditions
xVals = np.linspace(bounds[0], bounds[1], num=spacialNumber, endpoint=False);


#Define Common Initial Conditions
centerDiscontinuity = [1 if i < (spacialNumber)/2 else 0 for i in range(spacialNumber)];
oddCenterDiscontinuity = [1 if i < spacialNumber/2 else -1 for i in range(spacialNumber)];
centerDiscontinuityZeroSides = [1 if(xVals[i] > 0.25 and xVals[i] < 0.75) else 0 for i in range(spacialNumber)];
randomInitialCondition = np.random.rand(spacialNumber);

#Define Initial Condition
yValsInitial = randomInitialCondition;



#Define Common Potentials
harmonicOscillatorPotential = [((1/2) * 1.0**2 * abs(xVals[i])**2.0) for i in range(spacialNumber)];
freeSpace = [0 for _ in range(spacialNumber)];
softDoubleWell = [(xVals[i]/2 - 3)**2 * (xVals[i]/2 + 3)**2 * (1/8) for i in range(spacialNumber)];
softDoubleWell2 = [-1/4.0 * xVals[i]**2 * 16 + 1/2.0 * xVals[i]**4 + 8 for i in range(spacialNumber)];
finiteWell = [0 if (np.abs(xVals[i]) <= 1) else 100 for i in range(spacialNumber)];

#Define Potential Function
potential = harmonicOscillatorPotential;



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   PACKING AND CALLING BLOCK   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Pack all the parameters into lists to simplify function calling
packedParametersStrings = [simulationName, splittingType, integrationScheme];
packedParametersScalars = [numStates, spacialNumber, bounds[0], bounds[1], dx, dt, finalTime, order, 
                        steppingProduct, cauchyConvergenceTolerence, edgeTolerance, frameRate, secondsOfGif];
packedParametersBoolean = [bCalculateEnergy, bMomentumStates, bAllowCaching, bWriteEigenfunctions, 
                        bWriteEnergies, bAdaptiveSteping, bParityBreaking, bEven, bEvolutionStaging, bPlot, bCreateGif, 
                        bCreatePhaseGif, bUserOutput, bTimingSingle, bTimingTotal, bReset];
packedParametersArrays = [yValsInitial, potential, timeDescritizations, times];


#Call function
statesData, energies = imaginaryEigen.imaginaryTimeEigen(packedParametersStrings, packedParametersScalars, packedParametersBoolean, packedParametersArrays);