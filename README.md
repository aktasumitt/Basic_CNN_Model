# Basic_CNN_Model:

## Introduction:

In this project, I aimed to train a model using basic CNN architecture to recognize the hand signs of letters in American Sign Language and predict the meaning of the sign.

## Dataset:
- I used the ASL dataset for this project, which consists of 26 labels (letters) with 1500 images per label.
- I randomly split the dataset into training, validation, and test sets with ratios of (0.7, 0.2, 0.1) respectively.
- And I divided them into mini-batches with a batch size of 100. 
- Since the sizes of the images were varied, I resized them to (128x128x1) and applied normalization.
- Link For More information and downloading dataset: https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-alphabets

## Train:

- I built the model with a basic CNN structure. I repeated the structure three times, which consists of a convolutional layer (3x3 kernel, padding=1), Batch Normalization, Dropout (0.2), and Maxpooling (2x2 kernel, stride=2). Each time, the image size was halved. 
- Then, I used linear blocks to gradually reduce the units to the size of the label. 
- I chose Adam optimizer with a learning rate of 0.001 and used CrossEntropyLoss as the loss function. I trained the model for 6 epochs.

## Results:
- After 6 epochs, the model achieved approximately 98.5% accuracy on both the training, validation, and test sets, with a loss value of 0.064. 


## Usage: 
- You can train the model by setting "TRAIN_MODEL" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Then you can predict the images placed in the Prediction folder by setting the "Load_Model" and "Prediction" values to "True" in the config file.


