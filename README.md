# Rice Crops Disease Detection Using DenseNet121 Model

## Overview
This project implements a DenseNet121-based deep learning model to detect diseases in rice crops with high accuracy. By fine-tuning the DenseNet121 architecture on a custom dataset, the model delivers reliable predictions, assisting in early disease detection and management in agriculture.

## Features
- Efficient detection of rice crop diseases using DenseNet121.
- Fine-tuned with custom preprocessing and training techniques.
- High accuracy, validated with comprehensive evaluation metrics.
- Includes prediction visualizations with confidence scores.

## Dataset
- **Source**: A curated dataset of rice crop images, including healthy and diseased samples.
- **Classes**: The dataset contains **9 classes**, each representing a specific disease or a healthy crop.
- **Preprocessing**: 
  - Images resized to 128x128 pixels.
  - Normalized pixel values to the range [0, 1].
  - Data augmentation techniques applied for better generalization.

## Methodology
1. **Data Preparation**:
   - Images loaded using `ImageDataGenerator` with augmentation and normalization.
   - Batch size of 32 for efficient training.

2. **Model Architecture**:
   - DenseNet121 pre-trained on ImageNet, with top layers excluded.
   - Custom layers added:
     - Global Average Pooling.
     - Dense layers for classification with a softmax activation.
   - Base layers frozen during initial training for transfer learning.

3. **Training**:
   - Model trained for 20 epochs using Adam optimizer and categorical cross-entropy loss.
   - Training performance visualized using loss and accuracy graphs.

4. **Evaluation**:
   - Achieved 98% accuracy on the test set.
   - Confusion matrix generated to visualize class-wise performance.

## Dependencies
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn

## Results
- **Accuracy**: High accuracy achieved after fine-tuning.
- **Evaluation Metrics**: High precision, recall, and F1-score across all classes.
- Confusion matrix confirms the model's reliability in classification.

## Visualizations
- **Training Performance**:
  - Accuracy and loss graphs plotted for better understanding.
- **Confusion Matrix**:
  - Displays class-wise performance of the model.
- **Sample Predictions**:
  - Grid of actual vs. predicted labels with confidence scores.
- **Bar and Line Graphs**:
  - Showcases overall model evaluation metrics.

## Future Improvements
- Extend dataset to include more diseases and regional variations.
- Deploy model in real-time applications (mobile or web).
- Integrate with IoT devices for on-field disease detection.

## Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset contributors for their valuable efforts.
- TensorFlow and Keras for their robust deep learning frameworks.
- Open-source community for tools and libraries enabling this work. 
