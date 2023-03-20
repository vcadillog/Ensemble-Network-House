# Ensemble-Network-House
Simple script to learn about Ensemble learning with Neural Networks in Pytorch.

To use the code download the jupyter notebook in your local machine or in Google Colab and execute every cell.

The code contains 3 neural networks in the utils/networks module to predict cost price of a house:
1. Network with Fully connected layers to make a regression with metadata.
2. Network with Convolutional layers to make a regression with image data
3. Network with an Ensemble model of fully connected and convolutional network to combine metadata and images of the associated house.

This work also contains saved models trained in KFolds with 5 folds and no folds but longer epochs.

The log data can be visualized in W&B project: [URL](https://wandb.ai/vcadillo/House?workspace=user-vcadillo)
Additionally you can found in the chart a training loop for a lower batch size (16) has not generalized well.

Training Loss with no KFold training:

<img src="https://github.com/vcadillog/Ensemble-Network-House/blob/main/images/Train_loss.png" width="505" height="265"/>

Validation Loss with no KFold training:

<img src=https://github.com/vcadillog/Ensemble-Network-House/blob/main/images/Validation_loss.png" width="505" height="265"/>

Training Loss trained in 5 folds.

<img src="https://github.com/vcadillog/Ensemble-Network-House/blob/main/images/Train_kfold_loss.png" width="505" height="265"/>

Validation Loss trained in 5 folds.

<img src="https://github.com/vcadillog/Ensemble-Network-House/blob/main/images/Validation_kfold_loss.png" width="505" height="265"/>



[1] House dataset extracted from: https://github.com/emanhamed/Houses-dataset
 
