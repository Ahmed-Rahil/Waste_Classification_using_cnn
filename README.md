# Waste Classification using CNN

This project demonstrates how to classify waste into different categories using a Convolutional Neural Network (CNN). The implementation is done in a Jupyter Notebook using TensorFlow and OpenCV libraries.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Visualization](#visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to install the required libraries. You can install them using the following commands:

```bash
pip install numpy pandas matplotlib opencv-python tensorflow tqdm
```

## Dataset

The dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data). It is divided into training and testing sets and should be organized in the following structure:

```
Waste_Classification_using_cnn/
├── dataset/
│   ├── TRAIN/
│   │   ├── Organic/
│   │   └── Recyclable/
│   └── TEST/
│       ├── Organic/
│       └── Recyclable/
```

## Usage

To run the notebook, open it in Jupyter Notebook or Jupyter Lab and execute the cells sequentially. The notebook includes the following steps:

- **Import Libraries**: Import necessary libraries such as NumPy, Pandas, Matplotlib, OpenCV, and TensorFlow.
- **Load Dataset**: Define the paths to the training and testing datasets.
- **Data Preprocessing**: Load and preprocess the images.
- **Model Building**: Build the CNN model using TensorFlow and Keras.
- **Model Training**: Train the model on the training dataset.
- **Model Evaluation**: Evaluate the model on the testing dataset.
- **Visualization**: Visualize the results using Matplotlib.

## Project Overview

- **Library Imports**:

  - Import libraries for numerical operations (`numpy`), data manipulation (`pandas`), data visualization (`matplotlib.pyplot`), image processing (`cv2`), and progress tracking (`tqdm`).
  - Suppress warnings using the `warnings` library.
  - Import TensorFlow and Keras modules for machine learning functionalities.

- **Dataset Path Definition**:

  - Define paths for training and testing datasets, specifying where the images will be loaded from.

- **Data Loading and Preparation**:

  - Initialize lists for images (`x_data`) and labels (`y_data`).
  - Loop through the training directory to read images, convert them from BGR to RGB format, and append the image data and corresponding label to their respective lists.
  - Create a DataFrame (`data`) to organize images and their labels.

- **Data Visualization**:
  - Use a pie chart to visualize the distribution of the image classes ('Organic' and 'Recyclable') using `matplotlib`.

## Model Architecture

- **Convolutional Layers**: Extract features from images using convolution operations.
- **Activation Functions**: Apply non-linearity using 'ReLU' (Rectified Linear Unit).
- **MaxPooling Layers**: Down-sample feature maps to reduce dimensionality.
- **Flatten Layer**: Convert 2D feature maps into a 1D feature vector.
- **Dense Layers**: Classify the features with dropout layers to prevent overfitting.
- **Final Activation Layer**: Use 'sigmoid' for binary classification.

## Model Compilation

- Compile the model with the optimizer (`adam`) and loss function (`binary_crossentropy`), tracking accuracy as a metric.

## Data Augmentation and Generators

- Use `ImageDataGenerator` to preprocess images by normalizing pixel values.
- Create training and testing data generators to load images, resize them, and prepare them for training and validation.

## Model Training

- Train the model using the `fit` method with the training generator and validate against the test generator for a specified number of epochs.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

```

```
