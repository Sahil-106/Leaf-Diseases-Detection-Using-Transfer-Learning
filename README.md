# Potato Leaf Disease Classification

## Abstract

Potatoes are among the most widely consumed vegetables globally, ranking as the fourth most consumed. This project addresses the increasing global demand for potatoes, particularly during the COVID-19 pandemic, while highlighting the significant impact of potato diseases on crop quality and quantity. The study introduces a deep learning approach, utilizing convolutional neural network (CNN) models and transfer learning techniques, to classify two types of potato plant diseases based on leaf conditions.

## Introduction

Agriculture is a vital sector, especially in countries like India, where the economy is heavily dependent on it. Protecting crops from diseases is crucial for ensuring a successful harvest. This project focuses on the identification and classification of diseases affecting potato plants, aiming to reduce the time for disease identification and enhance precision in disease classification.

## Related Works

Several research studies have explored the use of deep learning algorithms and transfer learning for the classification of potato leaf diseases. Notable methods include CNN architectures such as Inception V3, VGG16, and Xception, demonstrating promising results in terms of accuracy.

## Dataset Description

The dataset used in this project consists of potato plant leaf images obtained from the 'Plant Village' database. The dataset includes healthy potato leaf images and images depicting two types of diseases: late blight and early blight. The dataset is divided into training and testing sets for model evaluation.

### Dataset Summary

- Healthy potato leaf: 152 images
- Early blight potato leaf: 999 images
- Late blight potato leaf: 1000 images

## Platform Utilized

The research and code execution were carried out on a machine with an AMD Ryzen 7 4800H CPU and an NVIDIA GeForce RTX 3050 GPU. The code was written in Python, utilizing libraries such as NumPy, Seaborn, Matplotlib, Sklearn, and TensorFlow. Transfer learning models, including VGG19, Xception, EfficientNetV2S, and DenseNet201, were employed.

## Proposed Model

### Transfer Learning

#### VGG19

- Total parameters: 21,073,475
- Trainable params: 1,049,091
- Non-trainable params: 20,024,384
- Training time: 22.45 min

#### Xception

- Total parameters: 25,056,299
- Trainable params: 4,194,819
- Non-trainable params: 20,861,480
- Training time: 12.25 min

#### EfficientNetV2S

- Total parameters: 22,953,315
- Trainable params: 2,621,955
- Non-trainable params: 20,331,360
- Training time: 19.20 min

#### Densenet201

- Total parameters: 22,254,659
- Trainable params: 3,932,675
- Non-trainable params: 18,321,984
- Training time: 38.28 min

### Deep Learning (CNN)

- Total parameters: 1,629,731
- Trainable params: 1,629,731
- Non-trainable params: 0
- Layers: Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

## Usage

To use the models for potato leaf disease classification, follow these steps:

1. Clone the repository: `git clone https://github.com/Sahil-106/Leaf-Diseases-Detection-Using-Transfer-Learning.git`
2. Navigate to the project directory: `cd Leaf-Diseases-Detection-Using-Transfer-Learning`

## Conclusion

After an in-depth study of various research papers on disease classification of potato plant diseases, this project proposes a model that leverages deep learning and transfer learning methods for the classification of potato plant leaf diseases. The CNN method is utilized for diseases classification, and various transfer learning models, including DenseNet, VGG19, EfficientNetV2S, and Xception, are employed to distinguish between diseased and non-diseased leaves.

### Transfer Learning Models Accuracies

| Framework       | Accuracy   |
| --------------- | ---------- |
| CNN             | 96.30%     |
| VGG19           | 92.00%     |
| Xception        | 85.58%     |
| EfficientNetV2S | 99.30%     |
| Densenet201     | 96.74%     |

### Transfer Learning Models Classification Metrics

#### 1. Densenet

| Disease Type  | Precision | Recall | F1 Score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Early Blight   | 0.97      | 0.98   | 0.98     | 203     |
| Late Blight    | 0.93      | 1.00   | 0.97     | 199     |
| Healthy        | 1.00      | 0.71   | 0.83     | 28      |

#### 2. EfficientNetV2S

| Disease Type  | Precision | Recall | F1 Score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Early Blight   | 1.00      | 1.00   | 1.00     | 203     |
| Late Blight    | 0.99      | 0.99   | 0.99     | 199     |
| Healthy        | 1.00      | 0.93   | 0.96     | 28      |

#### 3. VGG19

| Disease Type  | Precision | Recall | F1 Score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Early Blight   | 0.96      | 1.00   | 0.98     | 203     |
| Late Blight    | 0.88      | 0.96   | 0.92     | 199     |
| Healthy        | 1.00      | 0.07   | 0.13     | 28      |

#### 4. Xception

| Disease Type  | Precision | Recall | F1 Score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Early Blight   | 0.90      | 0.92   | 0.91     | 203     |
| Late Blight    | 0.81      | 0.91   | 0.86     | 199     |
| Healthy        | 0.00      | 0.00   | 0.00     | 28      |

The results indicate that EfficientNetV2S and Densenet201 models achieved the highest accuracy, with EfficientNetV2S reaching an impressive 99%. In conclusion, deep learning, specifically the CNN model, provides high accuracy, and among transfer learning models, EfficientNetV2S stands out for its superior classification performance.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to researchers and contributors in the field of agricultural image analysis.
- The 'Plant Village' database for providing the dataset.

Feel free to contribute to the project by forking and submitting pull requests!

