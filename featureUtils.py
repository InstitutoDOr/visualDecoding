import os;
import pickle;
import numpy as np;
from varUtils import img_dir;
from DNNUtils import makeCoders, transformFile, prepareNetwork2, createTransform;

def normalizeData2(matrix):
    m = np.mean(matrix, axis=0) # array([16.25, 26.25])
    std = np.std(matrix, axis=0) # array([17.45530005, 22.18529919])
    
    data = 0.5 * (np.tanh(0.01 * ((matrix - m) / std)) + 1);
    return data;

def normalizeData(matrix, percentile=1):
    dados = np.copy(matrix);
    m_neg = -3/np.percentile(dados, percentile);
    m_pos = 3/np.percentile(dados, 100-percentile);
    for i in range(dados.shape[1]):
        column = matrix[:, i];
        for j in range(len(column)):
            if column[j] < 0:
                column[j] = np.tanh(column[j] * m_neg);
            else:
                column[j] = np.tanh(column[j] * m_pos);
        dados[:, i] = column;
    return dados;    


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def selectBestVoxels(x_train, y_train, n_voxel=1000, voxel_selected = 1000):
    mask = np.zeros((x_train.shape[1],));
    n_unit = y_train.shape[1];
    
    corr = np.abs(corr2_coeff(x_train.T, y_train.T));

    for i in range(n_unit):
        correlationRank = np.argsort(corr[:, i])[::-1];
        voxel_index = correlationRank[n_voxel];
        mask[voxel_index] += 1;
        
    voxelRank = np.argsort(mask)[::-1];
    voxelRank = voxelRank[:voxel_selected];
    return voxelRank;
    
def buildSavedFeatures(tipo = 2):
    train_dir = '%snatural-image_training' % img_dir;
    files = os.listdir(train_dir);
    for tipo in range(6, 7):
        saveDir = 'savedFeatures/type%d' % tipo;
        if not os.path.exists(saveDir):
            os.makedirs(saveDir);
        featuresDict = {};
        if tipo == 1:
           _, encoder, _ = makeCoders();
        elif tipo == 2:
           _, encoder, _ = makeCoders(0);
        else:
           encoder = prepareNetwork2(tipo);
        transform = createTransform(tipo); 
        for file in files:
            filename = '%s/%s' % (train_dir, file);
            sample = transformFile(transform, filename);
            featuresDict[file] = encoder(sample).cpu().data.numpy();
            
        with open('%s/featuresType%d.pkl' % (saveDir, tipo), 'wb') as f:
             pickle.dump(featuresDict, f);

