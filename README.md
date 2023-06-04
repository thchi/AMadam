# 2023
######## VGG19_bn with different optimzer on cifar10 #######

! python /content/AMadam/cifar.py -a vgg19_bn --depth 20 --epochs 250 --optimizer  adam --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/vgg

! python /content/AMadam/cifar.py -a vgg19_bn --depth 20 --epochs 250 --optimizer  adamax --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/vgg

! python /content/AMadam/cifar.py -a vgg19_bn --depth 20 --epochs 250 --optimizer  adadelta --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/vgg


! python /content/AMadam/cifar.py -a vgg19_bn --depth 20 --epochs 250 --optimizer  amadam --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/vgg

! python /content/AMadam/cifar.py -a vgg19_bn --depth 20 --epochs 250 --optimizer  sgd --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/vgg




######## alexnet with different optimzer on cifar10 #######


! python /content/AMadam/cifar.py -a alexnet --depth 20 --epochs 250 --optimizer  adam --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/alex

! python /content/AMadam/cifar.py -a alexnet --depth 20 --epochs 250 --optimizer  adamax --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/alex

! python /content/AMadam/cifar.py -a alexnet --depth 20 --epochs 250 --optimizer  adadelta --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/alex

! python /content/AMadam/cifar.py -a alexnet --depth 20 --epochs 250 --optimizer  amadam --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/alex

! python /content/AMadam/cifar.py -a alexnet --depth 20 --epochs 250 --optimizer  sgd --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/alex
