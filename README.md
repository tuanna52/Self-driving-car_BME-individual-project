# BME individual project - NEAT based self-driving car
This repository is an individual project made at BME with the topic of self-driving car simulator and control algorithm. The code used in this project based originally on the [Neural_Network_NEAT](https://github.com/Hilicot/Neural_Network_NEAT) repo by Hilicot. I have made some changes (number of sensors, neural network architecture) and added some features to the code (draw statistical graph of the training process, saving the best genome after training, add new python script to run and test the performance of the saved genome).

Some main dependencies to run the code: NEAT, pygame, matplotlib, numpy and scipy.

To train the autonomous driving car, run main.py in python 3.7. After the training finished, there will be two output files: avg_fitness.svg and sd_car1.pkl. The first file contains the statistical graph of the training process, while the second one has the best genome architecture saved.

To run the trained model after saving it from the training process, run best_car_evaluation.py in python 3.7.

To tweak the settings without modifying the code, you can do it in config_variables.py and config_file.txt.

![image](https://user-images.githubusercontent.com/35283897/145893814-bd6b8d0f-52ba-4114-b94b-b6e4ecc5ec9b.png)
