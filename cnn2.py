import torch;
import torch.nn as nn;
import torch.utils.data as Data;


def run_decode_cnn(decode_id, x_train, y_train, x_test, y_test, minValue = -10.0, maxValue = 10.0):
    input_size = x_train.shape[1];
    output_size = y_train.shape[1];
    x_train = torch.from_numpy(x_train).float();
    x_test = torch.from_numpy(x_test).float();
  
    if 0:
        y_train = torch.from_numpy(y_train).float();
        y_test = torch.from_numpy(y_test).float();
            
    print('-------- training DNN model --------')
    for i in range(len(x_train)):
        x_train[i].requires_grad = True
        y_train[i].requires_grad = True
    
    traindata = Data.TensorDataset(x_train, y_train)
    
    if 0:
        class transnet(nn.Module):
            def __init__(self):
                super(transnet, self).__init__()
    
                self.hidden_layer01 = nn.Sequential(nn.Linear(input_size, 2048), nn.ReLU()); # melhorcom ReLu
                self.hidden_layer02 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU());
                self.hidden_layer03 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU());
                self.hidden_layer04 = nn.Sequential(nn.Linear(256, 128), nn.ReLU());
                self.hidden_layer05 = nn.Sequential(nn.Linear(128, 256), nn.ReLU());
                self.hidden_layer06 = nn.Sequential(nn.Linear(256, 1024), nn.ReLU());
                self.hidden_layer07 = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU());
                #self.output =  nn.Sequential(nn.Linear(2048, output_size), nn.Hardtanh(minValue, maxValue));
                self.output =  nn.Linear(2048, output_size);
            def forward(self, input):
                x = self.hidden_layer01(input);
                x = self.hidden_layer02(x);
                x = self.hidden_layer03(x);
                x = self.hidden_layer04(x);
                x = self.hidden_layer05(x);
                x = self.hidden_layer06(x);
                x = self.hidden_layer07(x);
                x = self.output(x);
                return x;
    elif 1:
        class transnet(nn.Module):
            def __init__(self):
                super(transnet, self).__init__()
    
                self.hidden_layer01 = nn.Sequential(nn.Linear(input_size, 2048), nn.ReLU()); # melhorcom ReLu
                self.hidden_layer02 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU());
                self.hidden_layer03 = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU());
                self.output =  nn.Sequential(nn.Linear(2048, output_size), nn.Hardtanh(minValue, maxValue));
                #self.output =  nn.Linear(2048, output_size);
            def forward(self, input):
                x = self.hidden_layer01(input);
                x = self.hidden_layer02(x);
                x = self.hidden_layer03(x);
                x = self.output(x);
                return x;
    elif 1:
        class transnet(nn.Module):
            def __init__(self):
                super(transnet, self).__init__()
    
                self.hidden_layer01 = nn.Sequential(nn.Linear(input_size, 2048), nn.Hardtanh(0.0, 9.0)); # melhorcom ReLu
                self.hidden_layer02 = nn.Sequential(nn.Linear(2048, 1024), nn.Hardtanh(0.0, 9.0));
                self.hidden_layer03 = nn.Sequential(nn.Linear(1024, 2048), nn.Hardtanh(0.0, 9.0));
                self.output =  nn.Sequential(nn.Linear(2048, output_size), nn.Hardtanh(minValue, maxValue));
            def forward(self, input):
                x = self.hidden_layer01(input);
                x = self.hidden_layer02(x);
                x = self.hidden_layer03(x);
                x = self.output(x);
                #x = 10 * torch.tanh(x);
                return x;
    elif 0:
        class transnet(nn.Module):
            def __init__(self):
                super(transnet, self).__init__()
    
                self.hidden_layer01 = nn.Sequential(nn.Linear(input_size, 1024), nn.ReLU()); # melhorcom ReLu
                self.hidden_layer02 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU());
                self.hidden_layer03 = nn.Sequential(nn.Linear(256, 1024), nn.ReLU());
                self.output =  nn.Sequential(nn.Linear(1024, output_size), nn.Hardtanh(minValue, maxValue));
            def forward(self, input):
                x = self.hidden_layer01(input);
                x = self.hidden_layer02(x);
                x = self.hidden_layer03(x);
                x = self.output(x);
                #x = 10 * torch.tanh(x);
                return x;
        
    elif 0:
        class transnet(nn.Module):
            def __init__(self):
                super(transnet, self).__init__()
    
                self.hidden_layer01 = nn.Sequential(nn.Linear(input_size, 2048), nn.Hardtanh(0.0, 9.0)); # melhorcom ReLu
                self.hidden_layer02 = nn.Sequential(nn.Linear(2048, 1024), nn.Hardtanh(0.0, 9.0));
                self.hidden_layer03 = nn.Sequential(nn.Linear(1024, 2048), nn.Hardtanh(0.0, 9.0));
                self.hidden_layer04 = nn.Sequential(nn.Linear(2048, 4096), nn.Hardtanh(0.0, 9.0));
                self.output =  nn.Sequential(nn.Linear(4096, output_size), nn.Hardtanh(minValue, maxValue));
            def forward(self, input):
                x = self.hidden_layer01(input);
                x = self.hidden_layer02(x);
                x = self.hidden_layer03(x);
                x = self.hidden_layer04(x);
                x = self.output(x);
                #x = 10 * torch.tanh(x);
                return x;
    else:
        class transnet(nn.Module):
            def __init__(self):
                super(transnet, self).__init__()
    
                self.hidden_layer01 = nn.Sequential(nn.Linear(input_size, 2048), nn.Hardtanh(0.0, 9.0)); # melhorcom ReLu
                self.hidden_layer02 = nn.Sequential(nn.Linear(2048, 1024), nn.Hardtanh(0.0, 9.0));
                self.output =  nn.Sequential(nn.Linear(1024, output_size), nn.Hardtanh(minValue, maxValue));
            def forward(self, input):
                x = self.hidden_layer01(input);
                x = self.hidden_layer02(x);
                x = self.output(x);
                #x = 10 * torch.tanh(x);
                return x;
            
    transNet = transnet().cuda()
    
    bat_size = 50;
    learning_rate = 1e-4;
    epochs = 300;
    threshold = 0.01;
    train_loader = Data.DataLoader(traindata, batch_size = bat_size, shuffle = True);
    optimizer = torch.optim.Adam(transNet.parameters(), lr = learning_rate);
    loss_function = nn.SmoothL1Loss();
    
    lastEpoch = 0;
    losses = [];
    for t in range(epochs):
        meanLoss = 0;
        for step, (x, y) in enumerate(train_loader):
            b_x = x.cuda();
            b_y = y.cuda();
            prediction = transNet(b_x);
            loss = loss_function(prediction, b_y);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        
            meanLoss +=  loss.cpu().data.numpy() * x.size(0);
        meanLoss = meanLoss / x_train.shape[0];
        lastEpoch = t+1;
        #print("train epoch: ", t, "loss", meanLoss);
        losses.append(meanLoss);
        if meanLoss < threshold:
           break; 
    print('Final Epoch: ', lastEpoch, ' Final loss : ', meanLoss);
    torch.save(transNet.state_dict(), 'c:\\Codigos\\Doutorado\\Reconstruction\\MyTest\\models\\dnnMapper_%s.pkl' % decode_id);
    
    train_pred = [];
    for i in range(x_train.shape[0]):
        prediction = transNet(x_train[i].cuda());
        train_pred.append(prediction);
        
    y_pred = [];
    for i in range(y_test.shape[0]):
        prediction = transNet(x_test[i].cuda());
        y_pred.append(prediction);
    
    print('-------- DNN training complete ----------')

    return y_pred, train_pred, transNet, losses;
