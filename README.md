# MBZUAI-ML703-Project

AdaptSegNet
train_gta2cityscapes_multi.py  train the model
evaluate_cityscapes.py         output the prediction results
compute_iou.py                 compute iou to evaluate the model

AdaIN-mnsit-svhn
create dataset.py  create and save the pictures of MNIST, SVHN
output.py          create and save the pictures whose style are transfered
svhn_test.py       train the model on the augmented source dataset and get the accuracy on the target dataset
net.py             encoder-decoder structure of AdaIN 
resnet.net         ResNet18 for training

DANN
config.py        write the position of the folder
model.py         structure of feature extractor and label classifier
train_source.py  train the source model
revgrad.py       use DANN to do domain adaptation
test_model.py    test the results of DANN on target dataset

CycleGAN-mnist-svhn
main.py          input basic parameters, train the cycleGAN, save the model and output pictures
data_loader      load the dataset
model.py         structure of 2 discriminators and 2 generators
solver.py        the procedure of training cycleGAN
