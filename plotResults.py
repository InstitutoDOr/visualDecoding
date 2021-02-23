from statsUtils import verifySignificance, verifyDifference2;
from plotUtils import plotResultsHuge2, plotResultsCC2, plotResultsCC3;
import matplotlib.pyplot as plt;


def plotComparison(excel5, excel7):
    plotResultsHuge2(excel7, excel5, 'correlationHuge.png', 89.75, 64.32, 93.16, 66.3);
    plt.pause(0.01);

    pKscore = 0;


    pscore = verifySignificance(excel7, 1, 1, Kamitani=True);
    pKscore = max(pKscore, pscore);
 
    pscore = verifySignificance(excel7, 1, 7, Kamitani=True);
    pKscore = max(pKscore, pscore);

    print('FC7');
    print('p-value KScore :', pKscore);
    
    pscore = verifySignificance(excel5, 1, 1, Kamitani=True);
    pKscore = max(pKscore, pscore);

    pscore = verifySignificance(excel5, 1, 7, Kamitani=True);
    pKscore = max(pKscore, pscore);

    print('Conv5');
    print('p-value KScore :', pKscore);
    

def plotCorrelationResults(excel5, excel7):
    plotResultsCC2(excel5, excel7, 'correlation_25088.png');
    plt.pause(0.01);
    
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('fc7 single data seen objects');
    tp1, tp5, tp10 = verifySignificance(excel7, 1, 1);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
    print('');
 
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('fc7 Mean data seen objects');
    tp1, tp5, tp10 = verifySignificance(excel7, 4, 1);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
    print('');
 
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('conv5 single data seen objects');
    tp1, tp5, tp10 = verifySignificance(excel5, 1, 1);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
    print('');
    
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('conv5 Mean data seen objects');
    tp1, tp5, tp10 = verifySignificance(excel5, 4, 1);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
    print('');
 
 
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('fc7 single data imagined objects');
    tp1, tp5, tp10 = verifySignificance(excel7, 1, 7);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
    print('');
 
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('fc7 Mean data imagined objects');
    tp1, tp5, tp10 = verifySignificance(excel7, 4, 7);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
    print('');
  
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('conv5 single data imagined objects');
    tp1, tp5, tp10 = verifySignificance(excel5, 1, 7);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
    print('');
  
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('conv5 Mean data imagined objects');
    tp1, tp5, tp10 = verifySignificance(excel5, 4, 7);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
 
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);


def plotPrototypeComparison(excel5, excel7):

    plotResultsCC3(excel5, excel7, 'correlation_25088.png');
    plt.pause(0.01);
    
    p1  = 0;
    p5  = 0;
    p10 = 0;
    
    print('');
    print('fc7 single data seen objects');
    tp1, tp5, tp10 = verifySignificance(excel7, 5, 1);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);

    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
 
    print('');
    print('fc7 Mean data seen objects');
    tp1, tp5, tp10 = verifySignificance(excel7, 4, 1);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);

    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
 
    print('');
    print('conv5 single data seen objects');
    tp1, tp5, tp10 = verifySignificance(excel5, 5, 1);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);

    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
    
    print('');
    print('conv5 Mean data seen objects');
    tp1, tp5, tp10 = verifySignificance(excel5, 4, 1);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
    
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
 
 
    p1  = 0;
    p5  = 0;
    p10 = 0;
    print('');
    print('fc7 single data imagined objects');
    tp1, tp5, tp10 = verifySignificance(excel7, 5, 7);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);

    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
 
    print('');
    print('fc7 Mean data imagined objects');
    tp1, tp5, tp10 = verifySignificance(excel7, 4, 7);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);

    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
  
    print('');
    print('conv5 single data imagined objects');
    tp1, tp5, tp10 = verifySignificance(excel5, 5, 7);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);

    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
    print('');
  
    print('');
    print('conv5 Mean data imagined objects');
    tp1, tp5, tp10 = verifySignificance(excel5, 4, 7);
    p1 = max(p1, tp1);
    p5 = max(p5, tp5);
    p10 = max(p10, tp10);
 
    print('p-value top 1 :', p1);
    print('p-value top 5 :', p5);
    print('p-value top 10:', p10);
