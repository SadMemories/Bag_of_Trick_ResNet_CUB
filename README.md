### Bag of Tricks for Image Classification with Convolutional Neural Networks
resnet50+CUB_200_2011

* 1: 未使用预训练权重，使用kaiming初始化，训练450 epochs，Top1-AC: 47.64%
* 2: 使用了ImageNet预训练权重，100 epochs，Top1-AC：66.45%
* 3: 将ResNet改为了ResNet-B(论文中有说明)的形式，100 epochs，Top1-AC：72.82%(使用预训练权重)
* 4: 将ResNet改为了ResNet-C(论文中有说明)的形式，100 epcohs，Top1-AC：55.99%
* 5: 将ResNet改为了ResNet-C(论文中有说明)的形式，450 epochs，Top1-AC：20.95%(未使用预训练权重)
