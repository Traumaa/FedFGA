#!bin/bash
MODEL=ResNet18_flanc

DEVICES=0

MODEL=ResNet18_flanc
CUDA_VISIBLE_DEVICES=$DEVICES python main_FL.py --n_agents 100 --dir_data ../ --data_train cifar10  --n_joined 10 --split iid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --project FLANC_CIFAR10 --template ResNet18 --model ${MODEL} --basis_fraction 0.125 --n_basis 0.25 --save FLANC  --dir_save ../experiment --save_models



MODEL=cnn_flanc
CUDA_VISIBLE_DEVICES=$DEVICES python main_FL.py --n_agents 100 --dir_data ../ --data_train fashion-mnist  --n_joined 10 --split iid --local_epochs 1 --batch_size 32 --epochs 200 --decay step-100 --lr 0.01 --fraction_list 0.25,0.5,0.75,1 --project FLANC --template ResNet18 --model ${MODEL} --basis_fraction 0.125 --n_basis 0.25 --save FLANC  --dir_save ../experiment --save_models


# IID cifar10
python main_FL.py --n_agents 100 --dir_data /home/user/dataset --data_train cifar10  --n_joined 10 --split iid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --project FLANC_CIFAR10 --template ResNet18 --model ResNet18_flanc --basis_fraction 0.125 --n_basis 0.25 --save FLANC  --dir_save /home/user/zhangxy/project/all-In-One-Neural-Composition-main/save --save_models

# NOIID cifar10
python main_FL.py --n_agents 100 --dir_data /home/user/dataset/ --data_train cifar10  --n_joined 10 --split noiid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --project FLANC_CIFAR10 --template ResNet18 --model ResNet18_flanc --basis_fraction 0.125 --n_basis 0.25 --save NOIID_cifi10

# IID cifar100
python main_FL.py --n_agents 100 --dir_data /home/user/dataset/ --data_train cifar100  --n_joined 10 --split iid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --project FL_CIFAR100 --template ResNet18 --model ResNet18_flanc --basis_fraction 0.125 --n_basis 0.25 --save IID_CIFAR100

# NOIID cifar100
python main_FL.py --n_agents 100 --dir_data /home/user/dataset/ --data_train cifar100  --n_joined 10 --split noiid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --project FL_CIFAR100 --template ResNet18 --model ResNet18_flanc --basis_fraction 0.125 --n_basis 0.25 --save NOIID_CIFAR100

# Fmnist IID
python main_FL.py --n_agents 100 --dir_data /home/user/dataset/ --data_train fashion-mnist  --n_joined 10 --split iid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.01 --fraction_list 0.25,0.5,0.75,1 --project FLANC_Fashion-Mnist --template common --model cnn_flanc --basis_fraction 0.125 --n_basis 0.25 --save IID_Fashion-Mnist

# Fmnist NOIID
python main_FL.py --n_agents 100 --dir_data /home/user/dataset/ --data_train fashion-mnist  --n_joined 10 --split noiid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.01 --fraction_list 0.25,0.5,0.75,1 --project FLANC_Fashion-Mnist --template common --model cnn_flanc --basis_fraction 0.125 --n_basis 0.25 --save NOIID_Fashion-Mnist

# vgg Fmnist IID
python main_FL.py --n_agents 100 --dir_data /home/user/dataset/ --data_train fashion-mnist  --n_joined 10 --split iid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.01 --fraction_list 0.25,0.5,0.75,1 --project FLANC_Fashion-Mnist --template vgg --model cnn_flanc --basis_fraction 0.125 --n_basis 0.25 --save VGG_IID_Fashion-Mnist

# vgg Fmnist NOIID
python main_FL.py --n_agents 100 --dir_data /home/user/dataset/ --data_train fashion-mnist  --n_joined 10 --split noiid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.01 --fraction_list 0.25,0.5,0.75,1 --project FLANC_Fashion-Mnist --template vgg --model cnn_flanc --basis_fraction 0.125 --n_basis 0.25 --save VGG_NOIID_Fashion-Mnist



# snn cifar10

python main_FL.py --n_agents 10 --dir_data /home/user/dataset --data_train cifar10 --data_test cifar10 --n_joined 10 --split iid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --project SNN_iidcf10 --template spiking_resnet --model spiking_resnet_flanc --basis_fraction 0.125 --n_basis 0.25 --save SNN_iidcf10

