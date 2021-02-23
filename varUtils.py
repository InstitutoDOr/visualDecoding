import json;

prototype_dir = 'prototypes';
#model_path = 'C:/codigos/Doutorado/ImageNet/imagenet/model.pth';
img_dir = 'stimuli/';
fmri_dir = "data";
resultDir = 'results';

class_info_json_filename = 'imagenet_class_info.json'
confusionMatrixDir = 'CMs/';

class_info_dict = dict()

with open(class_info_json_filename) as class_info_json_f:
    class_info_dict = json.load(class_info_json_f);


threshold = 10 * 1024;
count_threshold = 50;
imageRank = 30;
