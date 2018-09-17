# MLPneumonia
Kaggle challenge: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

## Pneumonia Localization

The pneumonia localization code uses the model trained by the pneumonia clasisification code as a fitness function for a genetic algorithm.  The genetic algorithm scans a cropping box over the xray, relying on the model to let it know when it has found a valid case of penumonia.  When it has it blacks out that region and tries again.  If it fails to find pneumonia after a specific amount of effort then it determines the xray to be penumonia free.

### GeneticAlgorithm.py

Generic genetic algorithm class.

### GeneticLocalization.py

Generic localization class.

### examine.py

Responsible for using find pnemonia in images and reporting the results
