# Graphics Card
  NVIDIA RTX 3050
# Training data:
	Grape: esca(3007 images), healthy(3061), leaf blight(3017)
	Mango: anthracnose(2240 images), healthy(2239), powdery mildew(1745)
	Tomato: bacterial spot(3955 images), healthy(2450 ), yellow leaf curl (4160)
# Testing data;
	Grape: esca(27 images), healthy(27), leaf blight(27)
	Mango: anthracnose(27 images), healthy(27), powdery mildew(27)
	Tomato: bacterial spot(27 images), healthy(27 ), yellow leaf curl (27)

# Models
  - plant_category: detects whether the leaf is either tomato, mango or grapes.
  - tomato_disease_category
  - mango_disease_category
  - grape_disease_category

# Setup
  - Updated the NVIDIA graphics drivers and installed NVIDIA CUDA toolkit.
  - Used the PyTorch domain library TorchVision to transform the data to tensors

# Model Architecture
      TinyVGG architecture
            3 convolutional blocks (3 convolutional layers, 2 max pool layers)
            1 classifier layer (Linear layer)
# Optimizer
      Adam 			
#Loss function
      Cross Entropy Loss
# Loss and Accuracy Plots

# Tran-Test Stats
## Plant Category Model
'''
Train loss: 0.020201700812664418 | Train accuracy: 0.9930970893141946 | Test loss: 0.011697725162024419 | Test accuracy: 0.998641304347826
100%|█████████████████████████████████████████████████████████████████████████| 30/30 [2:20:53<00:00, 281.77s/it] 
Total train time:  8453.208 seconds
'''

## Grape
'''
Train loss: 0.006556081525932506 | Train accuracy: 0.9976892605633803 | Test loss: 0.009489756424954976 | Test accuracy: 0.9921875
100%|████████████████████████████████████████████████████████████████████████████| 40/40 [14:21<00:00, 21.54s/it] 
Total train time:  861.635 seconds
'''

## Mango
'''
Train loss: 0.08840405036492321 | Train accuracy: 0.96875 | Test loss: 0.0599940475076437 | Test accuracy: 0.984375
100%|████████████████████████████████████████████████████████████| 40/40 [50:55<00:00, 76.39s/it] 
Total train time:  3055.625 seconds
'''

## Tomato 
'''
Train loss: 0.03834752611808192 | Train accuracy: 0.9861216012084593 | Test loss: 0.09959235228598118 | Test accuracy: 0.9677083333333334
100%|████████████████████████████████████████████████████████████| 40/40 [12:01<00:00, 18.04s/it] 
Total train time:  721.743 seconds
'''
