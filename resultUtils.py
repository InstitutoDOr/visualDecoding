import os;
import shutil;

import numpy as np;
from scipy.special import softmax;
import torch;
import json;
import pickle;
from listUtils import saveList, loadList;
from bdpy.stats import corrmat;
from varUtils import resultDir, img_dir, prototype_dir;
import pandas as pd;
import matplotlib.pyplot as plt;




imageRank = 30;

class_info_json_filename = 'imagenet_class_info.json'

class_info_dict = dict()

with open(class_info_json_filename) as class_info_json_f:
    class_info_dict = json.load(class_info_json_f);

def plotBars(filename, ylim=4600):
    if 'AverageMatrix' in filename:
        for nsubj in range(1, 6):
            subj = 'Subject%d' % nsubj;
            fname = filename.replace('AverageMatrix', subj);
            with open(fname + '_Ranks.pkl', 'rb') as f:
                 ranks_temp = pickle.load(f);
            if nsubj == 1:
                ranks = ranks_temp;
            else:
                for wnid in ranks_temp.keys():
                    ranks[wnid].extend(ranks_temp[wnid]);
            
    else:
        with open(filename + '_Ranks.pkl', 'rb') as f:
             ranks = pickle.load(f);
    
    top1 = 0;
    top5 = 0;
    top10 = 0;
    count = 0;
    positionList = [];
 
    for wnid in ranks.keys():
        class_top1 = 0;
        class_top5 = 0;
        class_top10 = 0;
        
        for rank in ranks[wnid]:
            count += 1;
            if wnid in rank[:1]:
                class_top1 +=1;
    
            if wnid in rank[:5]:
                class_top5 +=1;
                
            if wnid in rank[:10]:
                class_top10 +=1;

            positionList.append(rank.index(wnid) + 1);
                
        top1 += class_top1;
        top5 += class_top5;
        top10 += class_top10;
        
        class_top1 = class_top1 * 100 / len(ranks[wnid]);
        class_top5 = class_top5 * 100 / len(ranks[wnid]);
        class_top10 = class_top10 * 100 / len(ranks[wnid]);

        if 0:
            print('Top  1 : %f' % class_top1);
            print('Top  5 : %f' % class_top5);
            print('Top 10 : %f' % class_top10);
            print(len(ranks[wnid]));
            
    top1  = (top1 * 100)  / count;
    top5  = (top5 * 100)  / count;
    top10 = (top10 * 100) / count;
    
    if 1:
        print(filename);
        print('Top  1 : %f' % top1);
        print('Top  5 : %f' % top5);
        print('Top 10 : %f' % top10);
        print(count);
    
    if ylim != 0:
       plt.ylim(0, ylim);
       
    plt.figure(figsize = (15, 5));
    plt.hist(positionList);
    plt.pause(0.02);
    
    return top1, top5, top10;


def matrixAccuracyRanks(filename):
    if 'AverageMatrix' in filename:
        for nsubj in range(1, 6):
            subj = 'Subject%d' % nsubj;
            fname = filename.replace('AverageMatrix', subj);
            with open(fname + '_Ranks.pkl', 'rb') as f:
                 ranks_temp = pickle.load(f);
            if nsubj == 1:
                ranks = ranks_temp;
            else:
                for wnid in ranks_temp.keys():
                    ranks[wnid].extend(ranks_temp[wnid]);
            
    else:
        with open(filename + '_Ranks.pkl', 'rb') as f:
             ranks = pickle.load(f);
    
    top1 = 0;
    top5 = 0;
    top10 = 0;
    count = 0;
    for wnid in ranks.keys():
        class_top1 = 0;
        class_top5 = 0;
        class_top10 = 0;
        
        positionList = [];
        for rank in ranks[wnid]:
            count += 1;
            if wnid in rank[:1]:
                class_top1 +=1;
    
            if wnid in rank[:5]:
                class_top5 +=1;
                
            if wnid in rank[:10]:
                class_top10 +=1;

            positionList.append(rank.index(wnid) + 1);
                
        top1 += class_top1;
        top5 += class_top5;
        top10 += class_top10;
        
        class_top1 = class_top1 * 100 / len(ranks[wnid]);
        class_top5 = class_top5 * 100 / len(ranks[wnid]);
        class_top10 = class_top10 * 100 / len(ranks[wnid]);

        if 1:
            print(wnid, class_info_dict[wnid]['class_name']);

            plt.ylim(0, 175);
            plt.hist(positionList);
            plt.pause(0.02);

            print('Top  1 : %f' % class_top1);
            print('Top  5 : %f' % class_top5);
            print('Top 10 : %f' % class_top10);
            print(len(ranks[wnid]));
            
            
    top1  = (top1 * 100)  / count;
    top5  = (top5 * 100)  / count;
    top10 = (top10 * 100) / count;
    
    if 1:
        print(filename);
        print('Top  1 : %f' % top1);
        print('Top  5 : %f' % top5);
        print('Top 10 : %f' % top10);
        print(count);
    
    return top1, top5, top10;
    
def matrixAccuracy(filename):
    with open(filename + '.pkl', 'rb') as f:
         confusionMatrix = pickle.load(f);
    
    wnids = loadList('%s/test.txt' % img_dir);
    dfCM = pd.DataFrame(confusionMatrix, index = wnids, columns = wnids);
    dfCM = 1-dfCM;
    
    top1 = 0;
    top5 = 0;
    top10 = 0;
    for wnid in wnids:
        sortedVector = np.argsort(dfCM.loc[wnid, :])[::-1];
        rank = [wnids[i] for i in sortedVector]
        if wnid in rank[:1]:
            top1 +=1;

        if wnid in rank[:5]:
            top5 +=1;
            
        if wnid in rank[:10]:
            top10 +=1;
            
    top1  = (top1 * 100)  / len(wnids);
    top5  = (top5 * 100)  / len(wnids);
    top10 = (top10 * 100) / len(wnids);
    
    if 1:
        print(filename);
        print('Top  1 : %f' % top1);
        print('Top  5 : %f' % top5);
        print('Top 10 : %f' % top10);
    
    return top1, top5, top10;

def saveImageResults(target, predicted, correlations, number):
    dir_prediction = '%s/%d' % (resultDir, number);
    if not os.path.exists(dir_prediction):
       os.makedirs(dir_prediction);
    train_dir = '%scams' % img_dir;
    shutil.copyfile('%s/%s' % (train_dir, target), '%s/00target.jpeg' % dir_prediction);
    if 0:
       for i in range(imageRank):
           shutil.copyfile(predicted[i], '%s/%.4d.jpeg' % (dir_prediction, (i+1)));

    filenames = open('%s/filenames.txt' % dir_prediction, 'wt+');
    filenames.write('');

    filenames.write('%s/%s\n' % (train_dir, target));
    for i in range(len(predicted)):
        filenames.write('%i - %s \t %3.2f\n' % ((i+1), predicted[i], correlations[i]));
    filenames.close();
    
   
def calculateAccuracies(decoder, classNames, labels, predictions, filtered = False, CMFile = None):
    err_Top1 = 0;
    err_Top5 = 0;
    err_Top10 = 0;
    k_score = 0;

    testLabels = [];
    if len(testLabels) == 0:
       for j in range(len(labels)):
           testLabel = labels[j].split('_')[0];
           if testLabel not in testLabels:
               testLabels.append(testLabel);
       if len(testLabels) not in [50, 150, 32]:
          print('Error');

    classes = classNames;
    num_classes = len(classNames);
    if filtered:
       num_classes = len(testLabels);
       classes = testLabels;
       
    if CMFile is not None:
        confusionMatrix = np.zeros((num_classes, num_classes));
       
    print('Numero de classes', num_classes);              
    for i in range(len(labels)):
        features = decoder(predictions[i].unsqueeze(0));
        obtido = softmax(features.cpu().data.numpy())[0];
        
        predictedClassIdx = obtido.argmax();
        predictedClass = classNames[predictedClassIdx];
        targetClass = labels[i].split('_')[0];
        
        classRank = np.argsort(obtido)[::-1];
        classNameRank = [classNames[j] for j in classRank];

        if filtered:
            idxs = [];      
            classNameRankFiltered = [];
            for k in range(len(classNameRank)):
                if classNameRank[k] in testLabels:
                   classNameRankFiltered.append(classNameRank[k]); 
                   idxs.append(k);
            classNameRank = classNameRankFiltered;
            classRank = classRank[idxs];
            
        position = classNameRank.index(targetClass) + 1;
        k_score += (num_classes-position) / (num_classes-1);
        if 0:
            print('Target Class', targetClass, ' position: ', position, 'rank : ', classNameRank);
            print('Rank: ', obtido[classRank]);
        
        
        if CMFile is not None:
            line = classes.index(targetClass);
            for j in range(len(classNameRank)):
                confusionMatrix[line, classes.index(classNameRank[j])] += j / num_classes;

        if predictedClass != targetClass:
            err_Top1 += 1;
        
        if targetClass not in classNameRank[:5]:
            err_Top5 += 1;

        if targetClass not in classNameRank[:10]:
            err_Top10 += 1;

    top1 = 100 * (1-err_Top1/len(labels));
    top5 = 100 * (1-err_Top5/len(labels));
    top10 = 100 * (1-err_Top10/len(labels));
    k_score = 100 * (k_score/len(labels));
    
    print('Top 1  Accuracy (Test) : %3.2f%c ' % (top1, '%' ));
    print('Top 5  Accuracy (Test) : %3.2f%c ' % (top5, '%' ));
    print('Top 10 Accuracy (Test) : %3.2f%c ' % (top10, '%' ));
    print('Kamitani Score  (Test) : %3.2f%c ' % (k_score, '%' ));
    print('');
    
    
    if CMFile is not None:
        saveList(classes, os.path.splitext(CMFile)[0] + '_classes.txt');
        denom = len(labels) / 50;
        confusionMatrix = confusionMatrix / denom;
        with open(os.path.splitext(CMFile)[0] + '.pkl', 'wb') as f:
             pickle.dump(confusionMatrix, f);
        
    
    return top1, top5, top10, k_score;

def displayAccs(ws, column, wsStart, title, decoder, classNames, lbl_train, train_pred, filtered = False, CMFile = None):
    top1, top5, top10, k_score = calculateAccuracies(decoder, classNames, lbl_train, train_pred, filtered, CMFile);
    print(title);
    print('Top 1  Accuracy (Test) : %3.2f%c ' % (top1, '%' ));
    print('Top 5  Accuracy (Test) : %3.2f%c ' % (top5, '%' ));
    print('Top 10 Accuracy (Test) : %3.2f%c ' % (top10, '%' ));
    print('Kamitani Score  (Test) : %3.2f%c ' % (k_score, '%' ));
    print('');

    ws[column+'%d' % (wsStart+1)] = top1;
    ws[column+'%d' % (wsStart+2)] = top5;
    ws[column+'%d' % (wsStart+3)] = top10;
    ws[column+'%d' % (wsStart+4)] = k_score;


def resultadoMatriz(corrMatrix, classes, lbl_test, testLabels = None, filtered = False, CMFile = None):
    num_classes = len(classes);
    
    if filtered:
        if testLabels is not None:
            num_classes = len(testLabels);

    top1=0;
    top5=0;
    top10=0;
    k_score=0;
    classDict = {};
    
    if CMFile is not None:
        confusionMatrix = np.zeros((num_classes, num_classes));

    for i in range(len(lbl_test)): 
        featureClass = lbl_test[i].split('_')[0]; 
        correlations = np.squeeze(corrMatrix[i, :]);
               
        rank = np.argsort(correlations)[::-1];
    
        classRank = [classes[j] for j in rank];
        if filtered:
            idxs = [];
            for i in range(len(classRank)):
                if classRank[i] in testLabels:
                    idxs.append(i);
            auxRank = [];
            for i in idxs:
                auxRank.append(classRank[i]);
            classRank = auxRank;
            rank = rank[idxs];

    
        position = classRank.index(featureClass) + 1;
        k_score += (num_classes-position) / (num_classes-1);
    
        if 0:
            nameRanks = [];
            for wnid in classRank[:10]:
                nameRanks.append(class_info_dict[wnid]['class_name']);
            
            print('Target Class     : ', class_info_dict[featureClass]['class_name'], ' position: ', position, 'rank : ', nameRanks);
            #print('Correlation Rank : ', correlations[rank[:10]]);
 
        if CMFile is not None:
            line = classes.index(featureClass);
            for j in range(len(classRank)):
                confusionMatrix[line, classes.index(classRank[j])] += j / num_classes;
                
        if featureClass == classRank[0]:
            top1 += 1;
        else:
            top1 += 0;
    
        if featureClass in classRank[:5]:
            top5 += 1;
        else:
            top5 += 0;
    
        if featureClass in classRank[:10]:
            top10 += 1;
        else:
            top10 += 0;
            
        if featureClass not in classDict:
            classDict[featureClass] = [];
            
        classDict[featureClass].append(classRank);

    if CMFile is not None:
        saveList(classes, os.path.splitext(CMFile)[0] + '_classes.txt');
        denom = len(lbl_test) / 50;
        confusionMatrix = confusionMatrix / denom;

        with open(os.path.splitext(CMFile)[0] + '.pkl', 'wb') as f:
             pickle.dump(confusionMatrix, f);

        with open(os.path.splitext(CMFile)[0] + '_Ranks.pkl', 'wb') as f:
             pickle.dump(classDict, f);
        
    return top1, top5, top10, k_score;

def retornaClassePrototipos(classes, prototipos, counts, feature, featureFile, testLabels, filtered = True, showMessages = 0):
    from scipy.stats.stats import pearsonr;
 
    featureClass = featureFile.split('_')[0]; 
    num_classes = len(classes);
    
    if filtered:
        if testLabels is not None:
            num_classes = len(testLabels);
            
        corrMatrix = np.concatenate((np.reshape(feature, (1, prototipos.shape[1])), prototipos), axis=0);
        correlations = np.corrcoef(corrMatrix);
        correlations = np.squeeze(correlations[0, 1:]);
               
    rank = np.argsort(correlations)[::-1];
    
    classRank = [classes[j] for j in rank];
    if filtered:
        idxs = [];
        for i in range(len(classRank)):
            if classRank[i] in testLabels:
                idxs.append(i);
        auxRank = [];
        for i in idxs:
            auxRank.append(classRank[i]);
        classRank = auxRank;
        rank = rank[idxs];

    position = classRank.index(featureClass) + 1;
    k_score = (num_classes-position) / (num_classes-1);
    
    if 0:
        nameRanks = [];
        for wnid in classRank[:10]:
            nameRanks.append(class_info_dict[wnid]['class_name']);
        
        print('Target Class     : ', class_info_dict[featureClass]['class_name'], ' position: ', position, 'rank : ', nameRanks);
        #print('Correlation Rank : ', correlations[rank[:10]]);
 
    if featureClass == classRank[0]:
        top1 = 1;
    else:
        top1 = 0;

    if featureClass in classRank[:5]:
        top5 = 1;
    else:
        top5 = 0;

    if featureClass in classRank[:10]:
        top10 = 1;
    else:
        top10 = 0;

    return top1, top5, top10, k_score;

def calculaAcuracia(pred_y, lbl_test, protoTYPE = 1, tipo = 1, givenPrototypes = None, givenClasses = None, CMFile = None, showMessages = 0):
    classes = loadList('%s/classes.txt' % img_dir);
    if protoTYPE == 1 or protoTYPE == 2 or protoTYPE == 12:
        if tipo == 1:
            savepath = '%s/prototypes_200_4096.pkl' % prototype_dir;
        else:
            savepath = '%s/prototypes_200_25088.pkl' % prototype_dir;
    
        with open(savepath, 'rb') as f:
             prototipos = pickle.load(f);
    
        savepath = '%s/counts.pkl' % prototype_dir;
    
        with open(savepath, 'rb') as f:
             counts = pickle.load(f);
  
    if protoTYPE == 12:
        classes = classes[:150];
        prototipos = prototipos[:150, :];
        counts = counts[:150];

    if protoTYPE == 2:
        classes = classes[150:];
        prototipos = prototipos[150:, :];
        counts = counts[150:];
        
    if protoTYPE == 3:
        if tipo == 1:
           classes = loadList('%s/image_prototypes_4096.txt' % prototype_dir);
           savepath = '%s/image_prototypes_4096.pkl' % prototype_dir;
        else:
           classes = loadList('%s/image_prototypes_25088.txt' % prototype_dir);
           savepath = '%s/image_prototypes_25088.pkl' % prototype_dir;
    
        with open(savepath, 'rb') as f:
             prototipos = pickle.load(f);
            
        counts = np.zeros((prototipos.shape[0],));
        counts += 1;

    if protoTYPE == 4:
        classes = givenClasses;
        prototipos = givenPrototypes;
        counts = np.zeros((prototipos.shape[0],));
        counts += 1;

    if protoTYPE == 5:
        classes = loadList('%s/huge/hugeList.txt' % prototype_dir);
        mean_classes = loadList('%s/classes.txt' % prototype_dir);
        if tipo == 1:
           hugeproto = '%s/huge/prototypes_4096.pkl'  % prototype_dir;
           meanproto = '%s/prototypes_200_4096.pkl' % prototype_dir;
        else:
           hugeproto = '%s/huge/prototypes_25088.pkl'  % prototype_dir;
           meanproto = '%s/prototypes_200_25088.pkl' % prototype_dir;
    
        with open(hugeproto, 'rb') as f:
             prototipos = pickle.load(f);

        with open(meanproto, 'rb') as f:
             mean_prototipos = pickle.load(f);
    
        hugecounts = '%s/huge/counts.pkl' % prototype_dir;
    
        with open(hugecounts, 'rb') as f:
             counts = pickle.load(f);

    
    if protoTYPE == 8:
        classes = classes[:150];
        prototipos = None;
        counts = None;
        
    top1_acc = 0;
    top5_acc = 0;
    top10_acc = 0;
    k_score_acc = 0;
    
    testLabels = [];
    for lbl in lbl_test:
        className = lbl.split('_')[0];
        if className not in testLabels:
           testLabels.append(className);
    
    print('Numero de classes', len(testLabels));
    if len(pred_y[0].shape) == 1:
        for i in range(len(pred_y)):
            pred_y[i] = torch.unsqueeze(pred_y[i], dim=0);
    
    predictions = torch.cat(pred_y);
    predictions = predictions.cpu().data.numpy();
    
    if protoTYPE == 5:
       mean_classes = mean_classes[150:];
       mean_prototipos = mean_prototipos[150:, :];
  
       prototipos = np.vstack([mean_prototipos, prototipos]);
       classes = mean_classes + classes;
        
    corrMatrix = corrmat(predictions, prototipos);
    top1_acc, top5_acc, top10_acc, k_score_acc = resultadoMatriz(corrMatrix, classes, lbl_test, testLabels, CMFile = CMFile);        
   
    top1_acc = 100 * top1_acc / len(lbl_test);
    top5_acc = 100 * top5_acc / len(lbl_test);
    top10_acc = 100 * top10_acc / len(lbl_test);
    k_score_acc = 100 * k_score_acc / len(lbl_test);
    
    print('Top 1    : %3.2f%c' % (top1_acc, '%')); 
    print('Top 5    : %3.2f%c' % (top5_acc, '%')); 
    print('Top 10   : %3.2f%c' % (top10_acc, '%')); 
    print('K_score  : %3.2f%c' % (k_score_acc, '%')); 
    
    return top1_acc, top5_acc, top10_acc, k_score_acc;
    

def getColumn(sbj):
    if sbj == 'Subject1':
        return 'B';
    if sbj == 'Subject2':
        return 'C';
    if sbj == 'Subject3':
        return 'D';
    if sbj == 'Subject4':
        return 'E';
    if sbj == 'Subject5':
        return 'F';
        
def printResults(ws, column, wsStart, text, pred_y, testLabels, protoTYPE = 1, tipo = 1, CMFile = None):
    print(text);
    top1_acc, top5_acc, top10_acc, k_score_acc = calculaAcuracia(pred_y, testLabels, protoTYPE, tipo = tipo, CMFile = CMFile);
    
    ws[column+'%d' % (wsStart+1)] = top1_acc;
    ws[column+'%d' % (wsStart+2)] = top5_acc;
    ws[column+'%d' % (wsStart+3)] = top10_acc;
    ws[column+'%d' % (wsStart+4)] = k_score_acc;
