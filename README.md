### Bag of Tricks for Image Classification with Convolutional Neural Networks
resnet50+CUB_200_2011

* 1: 未使用预训练权重，使用kaiming初始化，训练450 epochs，Top1-AC: 47.64%
* 2: 使用了ImageNet预训练权重，100 epochs，Top1-AC：66.45%
* 3: 将ResNet改为了ResNet-B(论文中有说明)的形式，100 epochs，Top1-AC：72.82%(使用预训练权重)
* 4: 将ResNet改为了ResNet-C(论文中有说明)的形式，100 epcohs，Top1-AC：55.99%
* 5: 将ResNet改为了ResNet-C(论文中有说明)的形式，450 epochs，Top1-AC：20.95%(未使用预训练权重)
* 6: ResNet-B+ColorJitter+centerCrop(test)，100epochs，Top1-AC：76.91%
* 7: ResNet-B+ColorJitter+Resize(test), 100epochs, Top1-AC：69.8%
* 8: ResNet-B+centerCrop(test), 100 epochs, Top1-AC:79.06%
* 9: ResNet-B+centerCrop(test), 100 epochs, ImageNet的mean和std, Top1-AC:79.1%
* 10: ResNet-B+centerCrop(test), 100 epochs, ImageNet的mean和std+NAG, Top1-AC: 79.32%
* 11: ResNet-B+centerCrop(test), 100 epochs, ImageNet的mean和std+NAG+warm up(5), Top1-AC: 78.75%
* 12: ResNet-B+centerCrop(test), 100 epochs, ImageNet的mean和std+warm up(5)+cosine, Top1-AC: 79.06%
* 13: ResNet-B+centerCrop(test), 100 epochs, ImageNet的mean和std+cosine, Top1-AC: 78.91%
* 14: ResNet-B+centerCrop(test), 100 epochs, ImageNet的mean和std+cosine+LabelSmooth(0.1), Top1-AC: 81.43%
* 15: ResNet-B+centerCrop(test), 100 epcohs, ImageNet的mean和std+NAG+LabelSmooth(0.1), Top1-AC: 80.69%

