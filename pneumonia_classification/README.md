# MLPneumonia
Kaggle challenge: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

## Pneumonia Classification

The pneumonia classifier is a model which will be trained to say "does this subpart of the whole xray contain penumonia". To do this, it will train on images of positive pneumonia extracted from the data set, as well as images of negative pneumonia extracted from the data set.  The Pneumonia Localization genetic algoritm will then use this model as its heuristic for finding the bounding boxes in the radiological images.

### data.py

Responsible for extracting the images of positive pneumonia and negative from the source data and resizing them to the proper training size.  This is trivial for positive penumonia, as the training set is properly labeled for that.  For negative penumonia it is a little bit trickier, as there are no bounding boxes for "healthy" lungs.

### model.py

Creates the model used for penumonia classification

### train.py

Responsible for training the model on the data.
