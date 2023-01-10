# federated-learning-master


This repository contains a deep learning system (DLS) that addresses the limitations of federated learning as described in the paper ____ , and as used the paper___. 

Currently, federated learning restricts its applications to the classical batch setting, and requires that the entire dataset be available prior to training. Secondly, due to the nature of how neural networks are trained during federated learning, it relies on the assumption that the data and its underlying distributions are static. Neither of these assumptions are true in typical training settings.

In this project, we approach federated learning with an incremental learning perspective, where at each interval, a new data characteristic is introduced, replacing an old one and thus introducing the issue of catastrophic forgetting, the tendency of a neural network to “forget” knowledge previously obtained as it is trained on new data. This DLS utilizes knowledge distillation and elastic weight consolidation in order to solve catastrophic forgetting, allowing neural networks to be trained on a continuous, dynamic data stream, while maintaining privacy and security considerations.

## Data

To gauge our method’s performance on classification tasks, we trained and evaluated it on the task of diabetic retinopathy detection, using the public retinopathy dataset from EyePACS. The dataset consists of 88,702 fundus images from 44,351 patients, each with a clinically-provided label indicating the presence/severity of diabetic retinopathy (see Figure 6a and Figure 3) The labels range from 0 to 4 where 0 indicates no DR, 1 indicates mild DR, 2 indicates moderate DR, 3 indicates severe DR, and 4 indicates proliferative DR. Furthermore, to get a sense of performance on regression tasks, we trained and evaluated our model on the task of bone age prediction, using the public Bone Age Assessment (BAA) dataset from RSNA. The dataset consisted of 12,611 X-rays from 12,611 patients, and each image was accompanied with a label representing the age of the child in months at the time the X-ray was obtained. For this task, we used the gender of the children, male and female as “classes”

## Requirements

## Installation

Proceed below to complete the installation. You may need to manually install a GPU-enabled version of PyTorch by following the instructions here.

1. `git clone https://github.com/jho925/federated-learning-master.git`
2. `cd federated-learning-master`
3. `pip3 install -r requirements.txt`

## Usage

To run an experiment, either 

`cd diabetic-retinopathy` or `cd bone-age`

Depending on which dataset you would like to test the DLS on.

Then, in each respective directory, run 

`python3 main.py`

to run the experiment, using the command-line arguments below for specification

## Command-Line Arguments

### Model

```
--model <model>
```
Where `<model>` is any torchvision model, prefixed by models. (default= “models.resnet18”)

### Seed

```
--seed <seed>
```

Where `<seed>` is any integer (default=random.randint(0, 9999999))

### Learning Rate

```
--lr <lr>
```

Where `<lr>` is the learning rate of the model (default=0.0001)

### Rounds

```
--rounds <rounds>
```

Where `<rounds>` is how many rounds of training are used (default=1)

### Batch Size

```
--batch_size <batch_size>
```

Where `<batch_size>` is the batch size of the model (default=32)

### Epochs

```
--epochs_per <epochs_per>
```

Where `<epochs_per>` is the number of epochs used in each round of training (default=10)

### Sites

```
--sites <sites>
```

Where `<sites>` is how many institutions the model will travel between during training default=1)

### Positive Percent

```
--positive_percent <positive_percent>
```

Where `<positive_percent>` is what percentage the data split between positive/negative diagnosis is (for diabetic retinapothy) (default=0.5)

### Switch Distribution

```
--switch_distribution <switch_distribution>
```

Where `<switch_distribution>` is yes or no, It indicates whether to flip the positive negative distribution between round, ie if `<positive_percent>` is 0.8, then the data would be 80% positive 20% negative for round 1, and then vice versa for round 2, and so on (default=”yes”)

### Distillation Loss

```
--distillation_loss <distillation_loss>
```

Where `<distillation_loss>` is yes or no, indicating whether to use distillation loss (default=”no”)

### Weighted Loss

```
--weighted_loss <weighted_loss>
```

Where `<weighted_loss>` is yes or no, indicating whether to use weighted loss (default=”no”)

### Model Save Path

```
--model_save_path <path>
```

Where `<paths>` is where the model is to be saved (default=”model.pth”)

### Class Incremental

```
--class_incremental <class_incremental>
```

Where `<class_incremental>` is yes or no, indicating whether the experiment should be executed according the class incremental data shift scenario described in our paper (default=”no”)

### AUC

```
--val_auc <val_auc>
```

Where `<val_auc>` is yes or no, indicati whether to choose the best model using the auc score (default=”no”)

## Experiments

Below attached are the commands you should run for each experiment in order to replicate our results. These experiments are highlighted in more detail in our paper.

### Centrally Hosted (Homogeneous): 
#### Diabetic Retinapothy:
```
python3 main.py --train_size 2000 -epochs_per 64 --val_auc yes
```
#### Bone Age:
```
python3 main.py --train_size 2000 -epochs_per 64 --val_auc yes
```

### RDD Homogeneous: 
#### Diabetic Retinapothy:
```
python3 main.py --rounds 4 --sites 4 --train_size 2000 -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes
```
#### Bone Age:
```
python3 main.py --rounds 4 --sites 4 --train_size 2000 -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes
```

### DD Heterogeneous: 
#### Diabetic Retinapothy:
```
python3 main.py --rounds 4 --sites 4 --train_size 2000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes
```
#### Bone Age:
```
python3 main.py --rounds 4 --sites 4 --train_size 2000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes
```

### SDD Heterogeneous: 
#### Diabetic Retinapothy:
```
python3 main.py --rounds 1 --sites 4 --train_size 8000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes
```
#### Bone Age:
```
python3 main.py --rounds 1 --sites 4 --train_size 8000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes
```

### RDD Class Incremental:
#### Diabetic Retinapothy:
```
python3 main.py --rounds 4 --sites 4 --train_size 8000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes --class_incremental yes
```
#### Bone Age:
```
python3 main.py --rounds 4 --sites 4 --train_size 8000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes --class_incremental yes
```

### RDD Reverse Class Incremental: 
#### Diabetic Retinapothy (NEED TO FIX NOT SURE HOW WE DID THIS):
```
python3 main.py --rounds 4 --sites 4 --train_size 8000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes --class_incremental yes
```
#### Bone Age:
NA

### SDD Class Incremental: 
#### Diabetic Retinapothy:
```
python3 main.py --rounds 1 --sites 4 --train_size 8000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes --class_incremental yes
```
#### Bone Age:
```
python3 main.py --rounds 1 --sites 4 --train_size 8000 --positive_percent 0.8 --switch_distribution yes -epochs_per 64 --distillation_loss yes --weighted_loss yes --val_auc yes --class_incremental yes
```
 
## License

Distributed under the MIT License. See license.txt for more information.
