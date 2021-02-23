from prototypeTrainModes import prototypeTrain, prototypeTrainHuge;
from varUtils import confusionMatrixDir, resultDir;
import os;
from plotResults import plotComparison, plotPrototypeComparison, plotCorrelationResults;

if not os.path.exists(confusionMatrixDir):
    os.makedirs(confusionMatrixDir);

if not os.path.exists(resultDir):
    os.makedirs(resultDir);

if 1:
    FC7Results   = prototypeTrain(1);
    Conv5Results = prototypeTrain(2);
    

if 1:
    FC7ResultsHuge   = prototypeTrainHuge(1);
    Conv5ResultsHuge = prototypeTrainHuge(2);


if 1:
    # plot Graphs
    plotCorrelationResults(Conv5Results, FC7Results);
    plotPrototypeComparison(Conv5Results, FC7Results);
    
    # plot Graphs
    plotComparison(Conv5ResultsHuge, FC7ResultsHuge);    
    