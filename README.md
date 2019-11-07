# TrafficSign_GAN_Benchmark

## Dataset
Within the dataset folder can be found a P2 sign dataset for training purpose.

![](https://github.com/VincieD/TrafficSign_GAN_Benchmark/blob/master/dataset/P2/IS11b_0.4898%20008024_1433_211_105_112_140317.jpg)

## Training

In order to train a different architectures of GAN please refer to train_final_64x64.py
Within this script you can set up the the hyperparametrs of training and choose the desired architecture.

## Evaluation

For evaluation purposes 1000 images will be generated every 100 epochs. At the end of the training the FID will be generated. Please consider the discussion about FID, since 1000 images are consider to be not sufficient.
