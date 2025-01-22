# Waste Classification using CNN

This project demonstrates how to classify waste into different categories using a Convolutional Neural Network (CNN). The project is implemented in a Jupyter Notebook and uses TensorFlow and OpenCV libraries.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Visualization](#visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to install the required libraries. You can install them using the following commands:

```bash
pip install opencv-python
pip install tensorflow
```

### Dataset

The dataset used in this project is divided into training and testing sets. The dataset should be organized in the following structure:

```
Waste_Classification_using_cnn/
├── dataset/
│ ├── TRAIN/
│ │ ├── Organic/
│ │ └── Recyclable/
│ └── TEST/
│ ├── Organic/
│ └── Recyclable/
```

## Usage

To run the notebook, open it in Jupyter Notebook or Jupyter Lab and execute the cells sequentially. The notebook includes the following steps:

Import Libraries: Import necessary libraries such as NumPy, Pandas, Matplotlib, OpenCV, and TensorFlow.
Load Dataset: Define the paths to the training and testing datasets.
Data Preprocessing: Load and preprocess the images.
Model Building: Build the CNN model using TensorFlow and Keras.
Model Training: Train the model on the training dataset.
Model Evaluation: Evaluate the model on the testing dataset.
Visualization: Visualize the results using Matplotlib.

## Model Architecture

The CNN model is built using TensorFlow and Keras. The architecture includes the following layers:

Convolutional Layers
MaxPooling Layers
Flatten Layer
Dense Layers
Dropout Layers
BatchNormalization Layers
Visualization
The notebook includes code to visualize the distribution of the dataset and the results of the model. For example, a pie chart is used to show the distribution of organic and recyclable waste in the dataset.

## Results

The results of the model are displayed in the notebook, including the accuracy and loss during training and evaluation. The notebook also includes visualizations of the model's predictions.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
