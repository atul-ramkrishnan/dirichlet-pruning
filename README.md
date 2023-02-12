# Dirichlet pruning for neural network compression.
## Why I think this is a better use case than "Neural network regression" -- In the first paper, let's take the transfer learning example, the target is a probability simplex as a result of the application of softmax. On testing the covariance, I found out that the covariance matrix was entirely negative. The Dirichlet model is trained on this output, i.e. the target probability simplex itself doesn't have positive covariance. 
## Whereas in Dirichlet pruning, the probability simplex can be anything. There is no such restriction. Note that due to the nature of the probability simplex, where the sum of the RVs have to be 1, it is expected that the covariance is mostly negative. But it is still "theoretically" possible for there to be positive covariance, unlike the previous paper.

### Lenet

#### 1. First train the model

``cd Lenet``

```
python lenet_network_pruning.py --early_stopping 30
```

Then provide the name of the trained mode

#### 2. To prune:

Example (choose your own model from step 1):
```
python lenet_network_pruning.py --method shapley --shap_method kernel --shap_sample_num 30 --path_checkpoint_load checkpoint/scratch/mnist/mnist_trainval_0.8_epo_376_acc_99.05 --load_file 0 --resume 1 --prune_bool 1
```

#### 3. To prune and retrain:
```
python lenet_network_pruning.py --method shapley --shap_method kernel --shap_sample_num 30 --path_checkpoint_load checkpoint/scratch/mnist/mnist_trainval_0.8_epo_376_acc_99.05 --load_file 0 --resume 1 --prune_bool 1 --retrain 1
```

#### 4. To retrain a pruned model:
```
python lenet_network_pruning.py --method shapley --shap_method kernel --shap_sample_num 30 --path_checkpoint_load checkpoint/scratch/mnist/mnist_trainval_0.8_epo_376_acc_99.05 --load_file 1 --resume 1 --prune_bool 0 --retrain 1
```

Notes:

The default early-stopping value is 500, for trying the code choose a lower value.
`--early_stopping 30`
Choose an architecture to prune:
`--arch 8,9,36,17`
Choose the number of shapley samples (the default is 3)
`--shap_sample_num 100`
The previously computed Shapley samples can be reused with load_file
`--load_file 1`





### VGG

Please find all the available flags in the `VGG/vgg_load.py`

For basic run:

``cd VGG``

#### 1. To train the base network run:

```
python vgg_load.py --resume 1 --prune_bool 1 --retrain_bool 0 --method shapley --shap_method kernel --shap_sample_num 3 --model checkpoint/ckpt_vgg16_94.34.t7
```

#### 2. To prune:

``python vgg_load.py --resume 1 --prune_bool 1 --retrain_bool 0 --method shapley --shap_method kernel --shap_sample_num 3``

#### 3. To prune and retrain:

For example, select fisher pruning:

``python vgg_load.py --resume 1 --prune_bool 1 --retrain_bool 1 --method shapley --shap_method kernel --shap_sample_num 3``

Choose an architecture to prune:
`--pruned_arch 34,34,60,60,70,101,97,88,95,85,86,67,61,55,55`
The previously computed Shapley samples can be reused with load_file
`--load_file 1`


### Resnet 50

cd Resnet50

#### To train:

python resnet50_run3_prune.py --data <imagenet_data_path> --pretrained 0 --prune 0 --train_bool 1

#### 2. To prune:

python resnet50_run3_prune.py --data <imagenet_data_path> --pretrained 1 --prune 1 --train_bool 1 

#### 3. To prune and retrain:

python resnet50_run3_prune.py --data <imagenet_data_path> --pretrained 1 --prune 1 --train_bool 1 


The previously computed Shapley samples can be reused with load_file
`--load_file 1`

