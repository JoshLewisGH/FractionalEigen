from genericpath import exists
import numpy as np
from scipy.fft import fft, ifft;
import matplotlib.pyplot as plt;
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import os.path;
from os import mkdir;
from shutil import rmtree;
import ast
import warnings
warnings.filterwarnings('ignore')


def imaginaryTimeEigen(packedParametersStrings, packedParametersScalars, packedParametersBoolean, packedParametersArrays):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #   Simulation Parameters Block   #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #Simulation Defining Parameters
    simulationName = packedParametersStrings[0];
    splittingType = packedParametersStrings[1];
    integrationScheme = packedParametersStrings[2];
    numStates = packedParametersScalars[0];
    bCalculateEnergy = packedParametersBoolean[0];
    bMomentumStates = packedParametersBoolean[1];
    bAllowCaching = packedParametersBoolean[2];
    bWriteEigenfunctions = packedParametersBoolean[3];
    bWriteEnergies = packedParametersBoolean[4];

    #Initial Conditions and Potential
    yValsInitial = packedParametersArrays[0];
    potential = np.array(packedParametersArrays[1]);

    #System Defining Parameters
    spacialNumber = packedParametersScalars[1];
    bounds = [packedParametersScalars[2], packedParametersScalars[3]];
    dx = packedParametersScalars[4];
    dt = packedParametersScalars[5];
    finalTime = packedParametersScalars[6];
    timeSteps = int(finalTime/dt);
    order = packedParametersScalars[7];

    #Adaptive Stepping Parameters
    bAdaptiveSteping = packedParametersBoolean[5];
    steppingProduct = packedParametersScalars[8];

    #Symmetry Breaking Strategies
    bParityBreaking = packedParametersBoolean[6];
    bEven = packedParametersBoolean[7];

    #Accuracy Enforcing Parameters
    cauchyConvergenceTolerence = packedParametersScalars[9];
    edgeTolerance = packedParametersScalars[10];

    #Timer Staging Parameters
    bEvolutionStaging = packedParametersBoolean[8];
    timeDescritizations = packedParametersArrays[2];
    times = packedParametersArrays[3];

    #Simulation Plotting Parameters
    bPlot = packedParametersBoolean[9];
    bCreateGif = packedParametersBoolean[10];
    bCreatePhaseGif = packedParametersBoolean[11];
    frameRate = packedParametersScalars[11];
    secondsOfGif = packedParametersScalars[12];

    #User Feedback Parameters
    bUserOutput = packedParametersBoolean[12];
    bTimingSingle = packedParametersBoolean[13];
    bTimingTotal = packedParametersBoolean[14];

    #Reset Parameter 
    bReset = packedParametersBoolean[15];     #(KEEP FALSE IF YOU DON'T WANT TO REMOVE DATA)




    #Defines data saving structure for simulation
    if(not exists("Eigenfunctions/")):
        mkdir("Eigenfunctions/");
    if(not exists("StatePlots/")):
        mkdir("StatePlots/");
    if(not exists("MiscData/")):
        mkdir("MiscData/");
    eigenDirectoryName = "Eigenfunctions/" + simulationName + "/";
    if(not exists(eigenDirectoryName)):
        mkdir(eigenDirectoryName);
    elif(bReset):
        rmtree(eigenDirectoryName);
        mkdir(eigenDirectoryName);
    plotDirectoryName = "StatePlots/" + simulationName + "/";
    if(not exists(plotDirectoryName)):
        mkdir(plotDirectoryName);
    elif(bReset):
        rmtree(plotDirectoryName);
        mkdir(plotDirectoryName);
    miscDataDirectoryName = "MiscData/" + simulationName + "/";
    if(not exists(miscDataDirectoryName)):
        mkdir(miscDataDirectoryName);
    elif(bReset):
        rmtree(miscDataDirectoryName);
        mkdir(miscDataDirectoryName);




    #There are a few types of splitting that we will impliment with the following details
    # We define the standard LI error as the L infinity error for a single propogating soliton in the NLS equation with 100 time steps
    # We define the standard computational time as the time required to simulate a single propogating soliton in the NLS equation with 100 time steps

    # LieTrotter - steps: 2, order: 1, Standard LI error: 0.0013, Standard CompTime: 3.19219
    # Strange - steps: 2, order: 2, Standard LI error: 0.00014, Standard CompTime: 6.21
    # BM116PRK - steps: 11, order: 6, Standard LI error: 3.7e-10, Standard CompTime: 34.5879

    #Operator Splitting Coefficients
    if(splittingType == "LieTrotter"):
        steps = 1;
        opASplitterCoeffs = [1.0];
        opBSplitterCoeffs = [1.0];

    if(splittingType == "Strang"):
        steps = 2;
        opASplitterCoeffs = [0.5, 0.5];
        opBSplitterCoeffs = [1.0, 0.0];
        
    if(splittingType == "BM116PRK"):
        steps = 11;
        opASplitterCoeffs = [0.0502627644003922, 0.413514300428344, 0.0450798897943977, -0.188054853819569, 0.541960678450780, -0.7255255585086898,
                            0.541960678450780, -0.188054853819569, 0.0450798897943977, 0.413514300428344, 0.0502627644003922];
        opBSplitterCoeffs = [0.148816447901042, -0.132385865767784, 0.067307604692185, 0.432666402578175, -0.016404589403618, -0.016404589403618,
                            0.432666402578175, 0.067307604692185, -0.132385865767784, 0.148816447901042, 0.0000000000000000];
        
    if(splittingType == "CPR86SS"):
        steps = 8;
        opASplitterCoeffs = np.array([0.0584500187773306420+0.0217141273080301709*1.0j, 0.123229569418374774-0.0402806787860161256*1.0j, 0.158045797047111041-0.0604410907390099589*1.0j, 
                            0.160274614757183543+0.0790076422169959136*1.0j, 0.160274614757183543+0.0790076422169959136*1.0j, 0.158045797047111041-0.0604410907390099589*1.0j, 
                            0.123229569418374774-0.0402806787860161256*1.0j, 0.0584500187773306420+0.0217141273080301709*1.0j]);
        opBSplitterCoeffs = np.array([0.116900037554661284+0.0434282546160603418*1.0j, 0.129559101282088263-0.123989612188092593*1.0j, 0.186532492812133818+0.00310743071007267520*1.0j,
                            0.134016736702233270+0.154907853723919152*1.0j, 0.186532492812133818+0.00310743071007267520*1.0j, 0.129559101282088263-0.123989612188092593*1.0j,
                            0.116900037554661284+0.0434282546160603418*1.0j, 0.000000000000000000+0.0000000000000000000*1.0j]);
    
    
    def integrate(xVals, yVals):
        totalPoints = len(yVals);
        if(integrationScheme == "Trapazoidal"):
            return np.trapz(yVals, x=xVals);
        if(integrationScheme == "Simpsons"):
            dx = xVals[1] - xVals[0];
            divisions = totalPoints // 2;
            additionalPoints = totalPoints % 2;
            assert(additionalPoints == 1);
            sum = 0;
            for division in range(divisions):
                sum = sum + dx/3 * (yVals[division * 2] + 4 * yVals[division * 2 + 1] + yVals[division * 2 + 2])
            return sum;
        if(integrationScheme == "Simpsons4"):
            dx = xVals[1] - xVals[0];
            divisions = totalPoints // 3;
            additionalPoints = totalPoints % 3;
            assert(additionalPoints == 1);
            sum = 0;
            for division in range(divisions):
                sum = sum + dx * (3/8) * (yVals[division * 3] + 3 * yVals[division * 3 + 1] + 3 * yVals[division * 3 + 2] + yVals[division * 3 + 3])
            return sum;
        if(integrationScheme == "Booles"):
            dx = xVals[1] - xVals[0];
            divisions = totalPoints // 4;
            additionalPoints = totalPoints % 4;
            assert(additionalPoints == 1);
            sum = 0;
            for division in range(divisions):
                sum = sum + dx * (2/45) * (7 * yVals[division * 4] + 32 * yVals[division * 4 + 1] + 12 * yVals[division * 4 + 2] + 32 * yVals[division * 4 + 3] + 7 * yVals[division * 4 + 4])
            return sum;
        if(integrationScheme == "LS5"):
            dx = xVals[1] - xVals[0];
            divisions = totalPoints // 4;
            additionalPoints = totalPoints % 4;
            assert(additionalPoints == 1);
            sum = 0;
            for division in range(divisions):
                sum = sum + dx * (4/105) * (11 * yVals[division * 4] + 26 * yVals[division * 4 + 1] + 31 * yVals[division * 4 + 2] + 26 * yVals[division * 4 + 3] + 11 * yVals[division * 4 + 4])
            return sum;
        if(integrationScheme == "LS7"):
            dx = xVals[1] - xVals[0];
            divisions = totalPoints // 6;
            additionalPoints = totalPoints % 6;
            assert(additionalPoints == 1);
            sum = 0;
            for division in range(divisions):
                sum = sum + dx * (1/770) * (268 * yVals[division * 6] + 933 * yVals[division * 6 + 1] + 786 * yVals[division * 6 + 2] + 646 * yVals[division * 6 + 3] + 786 * yVals[division * 6 + 4] + 933 * yVals[division * 6 + 5] + 268 * yVals[division * 6 + 6])
            return sum;
        return -1;


    #Constructs an additional tag that is "tagged" onto the end of many output files to match symmetry breaking strategies
    tag = "";
    if(bParityBreaking and bEven):
        tag = "Even";
    if(bParityBreaking and not bEven):
        tag = "Odd";

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #   START OF SYSTEM INITIALIZATION   #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
   
   
    #Define position grid
    xVals = np.array(np.linspace(bounds[0], bounds[1], num=spacialNumber, endpoint=False));
    xValsPeriodic = np.array(np.append(xVals, bounds[1]));

    #Frequency Domain Grid Initialization 
    kVals = 2 * np.pi * np.fft.fftfreq(n=spacialNumber, d=dx);
    kVals = np.array(np.fft.fftshift(kVals));
    
    #Time step defining
    timeStep = dt;


    #Plot Initial Function
    initialFunc = [yValsInitial[i] for i in range(spacialNumber)];
    plt.figure(-1);
    plt.plot(xVals,initialFunc);
    plt.xlim(bounds);
    plt.ylim([min(initialFunc) - 0.1, max(initialFunc)*(1+0.1)]);
    plt.savefig(os.path.join(miscDataDirectoryName, "InitialFunc.png"));
    plt.close();

    #Plot Potential
    plt.figure(-1);
    plt.plot(xVals,potential);
    plt.xlim(bounds);
    plt.ylim([min(potential), max(potential)*(1+0.1)]);
    plt.savefig(os.path.join(miscDataDirectoryName, "Potential.png"));
    plt.close();


    #Defining Gif Parameters
    if(bCreateGif):
        frames = np.linspace(0, timeSteps - 1, frameRate*secondsOfGif);
        frames = [int(frame+0.5) for frame in frames]

    #Defining Plotting Labels
    labelPlot = "Spatial Order: " + str(round(order,4))

    #Initialization of storage variables
    statesData = [];
    momStatesData = [];
    frameData = [];
    energies = [];
    
    #Initialization of Plotting Variables
    mostMaxState = 0;
    mostMinState = 0;
    mostMaxProb = 0;
    mostMinProb = 0;

    #Timing Initialization and User Output
    if(bTimingTotal):
        initialStart = time.time();
    if(bUserOutput):
        print("Order of Simulation:",order);
        
    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #   END OF SYSTEM INITIALIZATION   #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    #This block does all the big stuff and each loop consists of a single state
    for stateNum in range(numStates):
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #   SINGLE STATE GENERATION   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        
        #Set timing
        if(bTimingSingle):
            singleStart = time.time();
        
        #User output
        if(bUserOutput):
            print("\nState number: " + str(stateNum));
        
        #Caching step: Checks if this eigenfunction has already been generated
        bAlreadyExists = False;
        if(bAllowCaching):
            fileName = "State" + str(stateNum) + "Order" + str(round(order,3)) + "Points" + str(spacialNumber) +  ".txt";
            pathName = os.path.join(eigenDirectoryName, fileName);
            if(exists(pathName)):
                bAlreadyExists = True;
                stateFile = open(pathName, "r");
                stateString = stateFile.read();
                stateString = stateString.replace("{","(").replace("}",")");
                stateList = ast.literal_eval(stateString);
                state = [stateList[index][1] for index in range(spacialNumber)];
                stateFile.close();
        
        
        #If the eigenfunction already exists there is no point to regenerate it so this part is skipped
        if(not bAlreadyExists):
            
            #Initialization
            state = np.array(yValsInitial);
            lastPercentPrinted = 0;
            
            #Imaginary Time Evolution Steps
            currentTime = 0;
            while currentTime < finalTime:
                
                #Output percent of time evolution completed
                currentPercent = (currentTime / finalTime) * 100;
                if(currentPercent // 10 > lastPercentPrinted and bUserOutput):
                    lastPercentPrinted = currentPercent // 10
                    print(str(lastPercentPrinted * 10) + "%");
                
                
                #Precompute using numpy vectorized options our evolution operators
                opA = np.exp(-1.0 * opASplitterCoeffs[:, np.newaxis] * potential * dt);
                opB = np.exp(1.0 * opBSplitterCoeffs[:, np.newaxis] * 0.5 * -(np.abs(kVals))**order * dt);
                
                #Time evolution
                tempState = state;
                priorState = state;
                for step in range(steps):
                    tempState = opA[step] * tempState;
                    tempState = fft(tempState);
                    tempState = np.fft.fftshift(tempState);                    
                    tempState = opB[step] * tempState;
                    tempState = np.fft.ifftshift(tempState);  
                    tempState = ifft(tempState);
                state = tempState;

                #Add to current time to reflect time evolution
                currentTime += dt;
                
                #Symmetry Breaking Step for Parity
                if(bParityBreaking):
                    if(spacialNumber % 2 == 0):
                        for j in range(int((spacialNumber)/2)):
                            state[spacialNumber - j - 1] = state[j + 1] if bEven else -1 * state[j + 1]
                    if(spacialNumber % 2 == 1):
                        for j in range(int((spacialNumber - 1)/2)):
                            state[spacialNumber - j - 1] = state[j + 1] if bEven else -1 * state[j + 1]
                            state[int((spacialNumber)/2)] = state[int((spacialNumber)/2)] if bEven else 0;
                
                #Cast to real plane to ensure uniqueness
                state = np.array(state)            
                
                #Include final point of state such that correct integration is done
                statePeriodic = np.append(state, state[0]);
                
                #Remove Prior States
                for energyState in statesData:
                    energyStatePeriodic = np.append(energyState, energyState[0]);
                    const = integrate(xValsPeriodic, energyStatePeriodic * statePeriodic)
                    state = state - const*energyState;
                
                #Renormalize Solution
                normalizationConstant = np.sqrt(np.abs(integrate(xValsPeriodic, statePeriodic * statePeriodic)))
                state = state/normalizationConstant
                
                #Change dt to better match the scaling of the hamiltonian better
                if(bAdaptiveSteping):
                    estimatedEnergy = -np.log(normalizationConstant)/dt;
                    dt = np.abs(steppingProduct/estimatedEnergy);
                    if(dt > timeStep):
                        dt = timeStep;
                        
                #Check cauchy criterion for convergence
                stateChange = np.abs(state - priorState)
                if(max(stateChange) < cauchyConvergenceTolerence):
                    if(bUserOutput):
                        print("Early Convergence Achieved")
                    break;
        
            #Tells user time evolution step is complete
            if(bUserOutput):
                print("100.0%\n\n")
            if(bTimingSingle and bUserOutput):
                print("Single state took " + str(time.time() - singleStart) + " seconds");
            
            
            #Additional Time Evolution
            if(bEvolutionStaging):
                for stageIndex in range(len(times)):
                    
                    print("Additional Evolution Stage:", stageIndex,"\n");
                    
                    dt = timeDescritizations[stageIndex];
                    additionalFinalTime = times[stageIndex];
                    
                    currentTime = 0;

                    while currentTime < additionalFinalTime:
                        #Precompute using numpy vectorized options for our evolution operators
                        opA = np.exp(-1.0 * opASplitterCoeffs[:, np.newaxis] * potential * dt);
                        opB = np.exp(1.0 * opBSplitterCoeffs[:, np.newaxis] * 0.5 * -(np.abs(kVals))**order * dt);
                        
                        #Time evolution
                        tempState = state;
                        priorState = state;
                        for step in range(steps):
                            tempState = opA[step] * tempState;
                            tempState = fft(tempState);
                            tempState = np.fft.fftshift(tempState);                    
                            tempState = opB[step] * tempState;
                            tempState = np.fft.ifftshift(tempState);  
                            tempState = ifft(tempState);
                        state = tempState;

                        #Add to current time to reflect time evolution
                        currentTime += dt;
                    
                        #Symmetry Breaking Step for Parity
                        if(bParityBreaking):                    
                            if(spacialNumber % 2 == 0):
                                for j in range(int((spacialNumber)/2)):
                                    state[spacialNumber - j - 1] = state[j + 1] if bEven else -1 * state[j + 1]
                            if(spacialNumber % 2 == 1):
                                for j in range(int((spacialNumber - 1)/2)):
                                    state[spacialNumber - j - 1] = state[j + 1] if bEven else -1 * state[j + 1]
                                    state[int((spacialNumber)/2)] = state[int((spacialNumber)/2)] if bEven else 0;                                
                        
                        #Cast to real plane to ensure uniqueness
                        state = np.array(state);
                        state = np.real(state)            
                        
                        #Include final point of state such that correct integration is done
                        statePeriodic = np.append(state, state[0]);
                        
                        #Remove Prior States
                        for energyState in statesData:
                            energyStatePeriodic = np.append(energyState, energyState[0]);
                            const = integrate(xValsPeriodic, np.conj(energyStatePeriodic) * statePeriodic)
                            state = state - const*energyState;
                        
                        #Renormalize Solution
                        normalizationConstant = np.sqrt(np.abs(integrate(xValsPeriodic, statePeriodic * np.conj(statePeriodic))))
                        state = state/normalizationConstant


            #Saves data recovered to states data
            statesData.append(state);
        else:            
            #Saves data recovered to states data
            statesData.append(state);
            print("State successfully cached...\n")
        
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #   POST STATE PROCESSING   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        
        #Calculate Energy From Decay of State
        if(bCalculateEnergy):
            
            #Define Time Evolution  operator Coefficents
            opATempSplitterCoeffs = [0.0584500187773306420+0.0217141273080301709*1.0j, 0.123229569418374774-0.0402806787860161256*1.0j, 0.158045797047111041-0.0604410907390099589*1.0j, 
                            0.160274614757183543+0.0790076422169959136*1.0j, 0.160274614757183543+0.0790076422169959136*1.0j, 0.158045797047111041-0.0604410907390099589*1.0j, 
                            0.123229569418374774-0.0402806787860161256*1.0j, 0.0584500187773306420+0.0217141273080301709*1.0j];
            opBTempSplitterCoeffs = [0.116900037554661284+0.0434282546160603418*1.0j, 0.129559101282088263-0.123989612188092593*1.0j, 0.186532492812133818+0.00310743071007267520*1.0j,
                            0.134016736702233270+0.154907853723919152*1.0j, 0.186532492812133818+0.00310743071007267520*1.0j, 0.129559101282088263-0.123989612188092593*1.0j,
                            0.116900037554661284+0.0434282546160603418*1.0j, 0.000000000000000000+0.0000000000000000000*1.0j];
            
            decayTime = dt;
            indexOfInterest = np.argmax(np.abs(state))
            tempState = state;
            for step in range(len(opASplitterCoeffs)):
                tempState = [np.exp(-1.0 * opATempSplitterCoeffs[step] * decayTime * potential[index]) * tempState[index] for index in range(spacialNumber)];
                tempState = fft(tempState);
                tempState = np.fft.fftshift(tempState);  
                tempState = [np.exp(1.0 * opBTempSplitterCoeffs[step] * decayTime * 0.5 * -(abs(kVals[index]))**order) * tempState[index] for index in range(spacialNumber)]; 
                tempState = np.fft.ifftshift(tempState);  
                tempState = ifft(tempState);
            
            energy = np.real(np.log(state[indexOfInterest]/tempState[indexOfInterest])/(decayTime));
            energies.append(energy);
            if(bUserOutput):
                print("Energy: ",energy);
                       
        
        #Plots recovered states
        if(bPlot):
            #Finds best maximum and minimum for plotting the states
            func = [np.real(state[i]) for i in range(spacialNumber)];
            if(max(func) > mostMaxState):
                mostMaxState = max(func);
            if(min(func) < mostMinState):
                mostMinState = min(func);
                
            #Finds best maximum and minimum for plotting the probability states
            propFunc = [np.real(state[i] * np.conj(state[i])) for i in range(spacialNumber)];
            if(max(propFunc) > mostMaxProb):
                mostMaxProb = max(propFunc);
            if(min(propFunc) < mostMinProb):
                mostMinProb = min(propFunc);

        
            #Plot the state function
            plt.figure(stateNum*2);
            plt.plot(xVals, func, label=labelPlot);
            plt.legend();
            plt.xlim(bounds);
            plt.ylim([mostMinState - 0.1, mostMaxState + 0.1]);
            fileName = "Order" + str(order).replace(".","_") + "State" + str(stateNum) + ".png";
            pathName = os.path.join(plotDirectoryName, fileName);
            plt.savefig(pathName);
            plt.close();

            #Plot the probability state function
            plt.figure(stateNum*2+1);
            plt.plot(xVals, propFunc, label=labelPlot);
            plt.legend();
            plt.xlim(bounds);
            plt.ylim([mostMinProb - 0.1, mostMaxProb + 0.1]);
            fileName = "Order" + str(order).replace(".","_") + "ProbState" + str(stateNum) + ".png";
            pathName = os.path.join(plotDirectoryName, fileName);
            plt.savefig(pathName);
            plt.close();


        #Creating gif step
        if(bCreateGif):
            fig, ax = plt.subplots()
            fig.set_tight_layout(True)

            x = np.linspace(bounds[0], bounds[1], spacialNumber);
            line, = ax.plot(x, [np.log10(np.abs(np.real(frameData[0][i]))) for i in range(spacialNumber)], 'r-', linewidth=2)
            #linePotential, = ax.plot(x, [xVals[i]**2/2 for i in range(spacialNumber)], 'r-', linewidth=2)


            def animate(frame):
                label = 'Gif Frame {0}'.format(frame)
                line.set_ydata([np.real(frameData[frame][i]) for i in range(spacialNumber)])
                #linePotential.set_ydata([xVals[i]**2/2 for i in range(spacialNumber)])
                ax.set_xlabel(label)
                ax.set_ylim([-8, 1.1])
                return line, ax;

            def init():
                line.set_ydata(np.ma.array(x, mask=True))
                return line,

            frames = [int(i) for i in np.linspace(0, len(frameData) - 1, secondsOfGif*frameRate)]

            figure = plt.figure(-20);
            anim_created = FuncAnimation(fig, animate, frames=frames, interval=25)

            writer = PillowWriter(fps=15)
            directoryName = miscDataDirectoryName;
            fileName = "Order" + str(order).replace(".","_") + "State" + str(stateNum) + ".gif";
            pathName = os.path.join(directoryName, fileName);
            anim_created.save(pathName, writer=writer);

    if(bWriteEnergies):
            output = str(energies);
            output = output.replace("e","*10^");
            output = output.replace("[","{");
            output = output.replace("]","}");
            fileName = "Energy" + str(numStates) + "Order" + str(round(order,9)) + "Points" + str(spacialNumber) + tag +  ".txt";
            pathName = os.path.join(miscDataDirectoryName, fileName);
            energyFile = open(pathName, "w");
            energyFile.write(output);
            energyFile.close();

    if(bTimingTotal):
        print("Total algorithm took " + str(time.time() - initialStart) + " seconds");



    #~~~~~~~~~~~~~~~~~~~#
    #   EIGEN WRITING   #
    #~~~~~~~~~~~~~~~~~~~#

    if(bWriteEigenfunctions):
        for stateIndex in range(len(statesData)):
            state = statesData[stateIndex];
            output = "{";
            for index in range(len(state)):
                output += "{" + str(round(xVals[index],6)) + "," + str(state[index]) + "},";
            output = output[:-1];
            output += "}";
            output = output.replace("e","*10^")
            if(bParityBreaking and bEven):
                stateIndex = stateIndex * 2;
            if(bParityBreaking and not bEven):
                stateIndex = stateIndex * 2 + 1;
            fileName = "State" + str(stateIndex) + "Order" + str(round(order,9)) + "Points" + str(spacialNumber) +  ".txt";
            pathName = os.path.join(eigenDirectoryName, fileName);
            eigenFile = open(pathName, "w");
            eigenFile.write(output);
            eigenFile.close();
        if(bMomentumStates):
            for stateIndex in range(len(momStatesData)):
                state = momStatesData[stateIndex];
                output = "{";
                for index in range(len(state)):
                    output += "{" + str(round(xVals[index],6)) + "," + str(state[index]) + "},";
                output = output[:-1];
                output += "}";
                fileName = "MomState" + str(stateIndex) + "Order" + str(round(order,3)) + "Points" + str(spacialNumber) +  ".txt";
                pathName = os.path.join(eigenDirectoryName, fileName);
                eigenFile = open(pathName, "w");
                eigenFile.write(output);
                eigenFile.close();

    return statesData, energies;