import os;
from bdpy import BData;
import numpy as np;
import sklearn.preprocessing;
from sklearn.preprocessing import MinMaxScaler, RobustScaler;

def id2Label(label):
    str_lbl = str(label).split('.');
    plh='';
    lbl_key = '';
    if len(str_lbl[0])==7:
       plh='0';
    str_idx = (str_lbl[1] + '0'*(6-len(str_lbl[1]))).lstrip('0');
    lbl_key = 'n'+plh+str_lbl[0]+'_' + str_idx;
    return lbl_key;

def id2File(label):
    return id2Label(label) + '.JPEG';

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

class data_handler():
    def __init__(self, filename):
        os.environ["H5PY_DEFAULT_READONLY"] = "1"
        self.bdata = BData(filename);
        self.dataTypeDict = {'train':1 , 'test':2 , 'test_imagine' : 3};

        img_ids = self.bdata.select('Label');
        datatype = np.squeeze(self.bdata.select('DataType'));
        
        self.trainIdxs = (datatype == self.dataTypeDict['train']);
        self.testIdxs = (datatype == self.dataTypeDict['test']);
        self.imagIdxs = (datatype == self.dataTypeDict['test_imagine']);

        img_ids_train = img_ids[self.trainIdxs, 0];
        img_ids_test = img_ids[self.testIdxs, 0];
        img_ids_imag = img_ids[self.imagIdxs, 0];

        self.train_labels  = [];
        self.test_labels  =  [];

        self.train_files  = [];
        self.test_files  =  [];
        self.imag_files = [];

        self.train_idxs  = [];
        self.test_idxs  =  [];
        self.imag_idxs = [];

        for id in img_ids_train:
            label = id2Label(id);
            if label not in self.train_labels:
               self.train_labels.append(label);

        for id in img_ids_test:
            label = id2Label(id);
            if label not in self.test_labels:
               self.test_labels.append(label);
               
        for id in img_ids_test:
            label = id2Label(id);
            self.test_idxs.append(self.test_labels.index(label));
            self.test_files.append(label + '.JPEG');

        for id in img_ids_train:
            label = id2Label(id);
            self.train_idxs.append(self.train_labels.index(label));
            self.train_files.append(label + '.JPEG');

        for id in img_ids_imag:
            label = id2Label(id);
            self.imag_idxs.append(self.test_labels.index(label));
            self.imag_files.append(label + '.JPEG');
            
        self.train_idxs = np.array(self.train_idxs);    
        self.test_idxs = np.array(self.test_idxs);    
        self.imag_idxs = np.array(self.imag_idxs);    
       


    def get_indices(self, imag_data = 0):

        if(imag_data):
            return self.train_idxs, self.test_idxs, self.imag_idxs;
        else:
            return self.train_idxs, self.test_idxs;



    def get_files(self, imag_data = 0):

        if(imag_data):
            return self.train_files, self.test_files, self.imag_files;
        else:
            return self.train_files, self.test_files;


    def get_data(self, normalize = 1, unityNormalization = 0, roi = 'ROI_VC', imag_data = 0):   # normalize 0-no, 1- per run , 2- train/test seperatly
        data = self.bdata.select(roi);

        if(normalize==1):
            xScaler = MinMaxScaler(feature_range = [-1, 1]);
            rScaler = RobustScaler(quantile_range = (1.0, 99.0));
            run = np.squeeze(self.bdata.select('Run').astype('int'))-1;
            data_norm = np.zeros(data.shape)

            # train
            for r in range(24):
                idxs = np.logical_and(r==run, self.trainIdxs);
                data_norm[idxs, :] = rScaler.fit_transform(data[idxs, :]);

                if unityNormalization==1:
                    data_norm[idxs, :] = xScaler.fit_transform( data_norm[idxs, :]);

            # test
            for r in range(35):
                idxs = np.logical_and(r==run, self.testIdxs);
                data_norm[idxs, :] = rScaler.fit_transform(data[idxs, :]);

                if unityNormalization==1:
                    data_norm[idxs, :] = xScaler.fit_transform( data_norm[idxs, :] );

            # test_imag
            for r in range(20):
                idxs = np.logical_and(r==run, self.imagIdxs);
                data_norm[idxs, :] = rScaler.fit_transform(data[idxs, :]);

                if unityNormalization==1:
                    data_norm[idxs, :] = xScaler.fit_transform( data_norm[idxs, :] );

                    
            train_data = data_norm[self.trainIdxs, :];
            test_data  = data_norm[self.testIdxs, :];
            test_imag = data_norm[self.imagIdxs, :];

        else:
            train_data = data[self.trainIdxs, :];
            test_data  = data[self.testIdxs, :];
            test_imag = data[self.imagIdxs, :];

            if(normalize==2):
                train_data = sklearn.preprocessing.scale(train_data);
                test_data = sklearn.preprocessing.scale(test_data);
                test_imag = sklearn.preprocessing.scale(test_imag);


        test_labels = self.test_idxs;
        imag_labels = self.imag_idxs;
        
        num_labels = max(test_labels) + 1;
        test_data_avg = np.zeros([num_labels, test_data.shape[1]])
        test_imag_avg = np.zeros([num_labels, test_data.shape[1]])

        for i in range(num_labels):
            test_data_avg[i, :] = np.mean(test_data[test_labels == i, :], axis=0);
            test_imag_avg[i, :] = np.mean(test_imag[imag_labels == i, :], axis=0);
            
        if(imag_data):
            return train_data, test_data, test_data_avg,test_imag,test_imag_avg

        else:
            return train_data, test_data, test_data_avg

