# Sanro Health - Diabetic Retinopathy Classifier

## Installation Steps

```bash
git clone https://github.com/davidwmcdevitt/SanroDR
%cd SanroDR
!pip install -r requirements.txt
```

## Data Preparation

```bash
unzip [TRAINING_DATA.ZIP] -d data
unzip [TEST_DATA.ZIP] -d test
unzip [MESSIDOR2.ZIP] -d test
```

## Replicate Training
```bash
python train.py [OPTIONS]

--data_dir : Specifies the directory where the training data is located. Default is data.
--num_epochs : Sets the number of training epochs. Default is 25.
--oversample : Enables oversampling of minority classes to balance the dataset.
--class_weights : Includes class weights to handle class imbalance during training.
--force_balance : Forces the dataset to have a 50/50 split between 'No DR' cases and 'any DR' cases, helping balance the training data.
--state_dict : Path to a saved model state dictionary for continuing training from a previously saved state.
```

## Evaluate Model
```bash
python eval.py [OPTIONS]

--eval_set : Specifies the dataset to be used for evaluation. Must be one of 'kaggle' or 'messidor2'.
--model_dict : Path to the model state dictionary file that you want to evaluate. 
--data_dir : The directory where the evaluation data is stored.
```
