import torchvision.models as models;
import torchvision.transforms as transforms;

import torch;
import torch.nn as nn;
import torch.nn.parallel;
import platform;
from PIL import Image;
import pickle;


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

if platform.system() == 'Windows':
    model_path = 'C:/codigos/Doutorado/ImageNet/imagenet/model.pth';
else:
    model_path = '/root/deepLearning/doutorado/model/model.pth';


def preparaNetwork(num_classes = 200):
    model_ft = models.vgg19_bn(pretrained=True);
    for param in model_ft.parameters():
        param.requires_grad = False;
        
    n_inputs = model_ft.classifier[6].in_features;    
    model_ft.classifier[6] = nn.Sequential(
                                nn.Linear(n_inputs, num_classes),
                                nn.LogSoftmax(dim=1)
                                );
    
    model_ft = model_ft.to(device);
    return model_ft;

def prepareNetwork2(tipo=3):
    if tipo in [3, 4, 6]:
        if tipo == 3:
           encoder = models.inception_v3(pretrained = True);
        elif tipo == 4:
           encoder = models.resnext101_32x8d(pretrained = True);
        elif tipo == 6:
           encoder = models.resnet152(pretrained = True);
        encoder.fc = nn.Sequential();
    elif tipo == 5:    
        encoder = models.densenet161(pretrained = True);
        encoder.classifier = nn.Sequential();
        
    encoder.eval();
    for param in encoder.parameters():
       param.requires_grad = False;
    encoder = encoder.cuda();
        
    return encoder;       
    
def loadModel():
    fname = model_path;
    model_ft = preparaNetwork();
    model_ft.load_state_dict(torch.load(fname));
    model_ft.eval();
    return model_ft;

def loadModel2():
    model_ft = models.vgg19_bn(pretrained=True);
    for param in model_ft.parameters():
        param.requires_grad = False;
        
    model_ft = model_ft.to(device);
    model_ft.eval();
    return model_ft;

def loadModel3():
    import pretrainedmodels;
    model_name = 'inceptionv4' # could be fbresnet152 or inceptionresnetv2
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet');
    for param in model.parameters():
        param.requires_grad = False;
    model.eval();
    model.cuda();
    return model;
    
def makeCoders(layerIndex=-3):
    classification = loadModel();

    encoder = loadModel();
    if layerIndex == 0:
        encoder.classifier = nn.Identity();
        decoder = nn.Sequential(*list(classification.classifier.children()));
    else:    
        encoder.classifier = nn.Sequential(*list(encoder.classifier.children())[:layerIndex]);
        decoder = nn.Sequential(*list(classification.classifier.children())[layerIndex:]);
    return classification, encoder, decoder;    

def makeCoders2(layerIndex=-3):
    classification = loadModel2();

    encoder = loadModel2();
    encoder.classifier = nn.Sequential(*list(encoder.classifier.children())[:layerIndex]);

    decoder = nn.Sequential(*list(classification.classifier.children())[layerIndex:]);
    return classification, encoder, decoder;    

def makeCoders3(layerIndex=-3):
    classification = loadModel2();

    encoder = loadModel2();
    encoder.classifier = nn.Identity();

    decoder = None;
    return classification, encoder, decoder;    

def makeCoders4(layerIndex=-3):
    classification = None;

    encoder = loadModel3();
    encoder.last_linear = nn.Identity();

    decoder = None;
    return classification, encoder, decoder;    

def transformFile(transform, arq):
    img = pil_loader(arq);
    sample = transform(img);
    sample = sample.unsqueeze(0).cuda();
    return sample;

def transformFile2(load_img, tf_img, arquivo):
    input_img = load_img(arquivo);
    input_tensor = tf_img(input_img);         # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
    input_vector = torch.autograd.Variable(input_tensor,requires_grad=False).cuda();
    return input_vector;
    
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f);
        return img.convert('RGB');


def createTransform(tipo=1):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if tipo in [1,2,4,5,6]:
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(), 
        								normalize]);
    elif tipo == 3:
        transform = transforms.Compose([transforms.Resize(299),
                                        transforms.CenterCrop(299),
                                        transforms.ToTensor(), 
        								normalize]);
    
    return transform;

    
def createFeatures(encoder, tipo, img_dir, lbl_train, lbl_test):
    transform = createTransform(tipo);

    train_dir = '%s/natural-image_training' % img_dir;
    featuresTrain = [];
    for i in range(len(lbl_train)):
        file = '%s/%s' % (train_dir, lbl_train[i]);
        sample = transformFile(transform, file);
        featuresTrain.append(encoder(sample));
    
    test_dir = '%s/natural-image_test' % img_dir;
    featuresTest = [];
    for i in range(len(lbl_test)):
        file = '%s/%s' % (test_dir, lbl_test[i]);
        sample = transformFile(transform, file);
        featuresTest.append(encoder(sample));
    return featuresTrain, featuresTest;

def createFeaturesFromDisk(encoder, tipo, img_dir, lbl_train, lbl_test):
    with open('savedFeatures/type%d/featuresType%d.pkl' % (tipo, tipo), 'rb') as f:
         featuresDict = pickle.load(f);
    featuresTrain = [];
    for i in range(len(lbl_train)):
        featuresTrain.append(torch.from_numpy(featuresDict[lbl_train[i]]).cuda());
    
    featuresTest = [];
    for i in range(len(lbl_test)):
        featuresTest.append(torch.from_numpy(featuresDict[lbl_test[i]]).cuda());

    return featuresTrain, featuresTest;

def createFeatures2(encoder, tipo, img_dir, lbl_train, lbl_test):
    transform = createTransform(tipo);

    featuresTrain = [];
    for i in range(len(lbl_train)):
        file = '%s/%s' % (img_dir, lbl_train[i]);
        sample = transformFile(transform, file);
        featuresTrain.append(encoder(sample));
    
    featuresTest = [];
    for i in range(len(lbl_test)):
        file = '%s/%s' % (img_dir, lbl_test[i]);
        sample = transformFile(transform, file);
        featuresTest.append(encoder(sample));
    return featuresTrain, featuresTest;

def createFeatures3(encoder, img_dir, lbl_train, lbl_test):
    import pretrainedmodels.utils as utils;
    load_img = utils.LoadImage();
    tf_img = utils.TransformImage(encoder);

    train_dir = '%s/natural-image_training' % img_dir;
    featuresTrain = [];
    
    for i in range(len(lbl_train)):
        file = '%s/%s' % (train_dir, lbl_train[i]);
        sample = transformFile2(load_img, tf_img, file);
        featuresTrain.append(encoder(sample));
    
    test_dir = '%s/natural-image_test' % img_dir;
    featuresTest = [];
    for i in range(len(lbl_test)):
        file = '%s/%s' % (test_dir, lbl_test[i]);
        sample = transformFile2(load_img, tf_img, file);
        featuresTest.append(encoder(sample));
    return featuresTrain, featuresTest;
