# SINAYA-A-Fuzzy-Neural-Network
A Fuzzy Neural Network Model aimed to infer the ammonia level based on the actual measured values of pH, Dissolved Oxygen, and Temperature.

# Ammonia Concentration Prediction using Fuzzy Neural Network

This repository contains code for training a neural network model that uses fuzzy logic to predict the concentration of ammonia based on pH, dissolved oxygen (DO), and temperature inputs.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

## Introduction

Ammonia is an important parameter to monitor in environmental and water quality assessment. This code provides a neural network model that incorporates fuzzy logic to predict ammonia concentration based on pH, DO, and temperature. The model uses fuzzy membership functions to capture the relationship between input variables and output. The code normalizes the data and trains the model using the Nadam optimizer and mean squared error loss.

## Installation

To run the code, you need to have the following dependencies installed:

- TensorFlow
- Keras
- NumPy
- Pandas
- scikit-learn

You can install these dependencies using pip:

```shell
pip install tensorflow keras numpy pandas scikit-learn
```

## Usage

1. Clone the repository:

```shell
git clone <repository-url>
```

Replace `<repository-url>` with the URL of this GitHub repository.

2. Navigate to the repository directory:

```shell
cd ammonia-concentration-prediction
```

3. Run the `ammonia_prediction.py` script:

```shell
python ammonia_prediction.py
```

This script loads the training data, normalizes the input and output, defines the fuzzy neural network model, trains the model, and saves it as an h5 file.

4. After running the script, the trained model will be saved as `SINAYA_FFNN_FINAL4.h5` in the current directory.

## Dataset

The training data is stored in a CSV file named "Training Dataset v2.csv". It should contain columns for pH, dissolved oxygen (DO), temperature, and ammonia concentration. Make sure the CSV file is located in the same directory as the code.

## Model Architecture

The neural network model consists of three input layers for pH, DO, and temperature, respectively. Fuzzy membership functions are applied to each input variable, and the fuzzy rules are defined using lambda layers. The fuzzy rules are concatenated, and the output is computed using a dense layer with linear activation.

The model architecture is as follows:

1. Input layer for pH
2. Input layer for DO
3. Input layer for temperature
4. Fuzzy membership functions for pH, DO, and temperature
5. Concatenation of fuzzy membership functions
6. Fuzzy rules
7. Concatenation of fuzzy rules
8. Output layer with linear activation

## Training

The model is trained using the Nadam optimizer with a learning rate of 0.001, beta parameters of 0.6 and 0.665, and epsilon of 7e-08. The loss function used is mean squared error (MSE), and the model is trained for 3000 epochs with a batch size of 64.

## Evaluation

The accuracy metric is used to evaluate the model during training. However, it's important to note that the accuracy value may not be the most suitable evaluation metric for regression problems. Additional evaluation methods, such as calculating mean absolute error (MAE) or root mean squared error (RMSE), can be used to assess the model's performance on unseen data.

## License

This code is released

 under the [MIT License](LICENSE).
```

Feel free to customize the README file according to your specific needs and add any additional sections or information you think would be relevant for users accessing your code repository.
