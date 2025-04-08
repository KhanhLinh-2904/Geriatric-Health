## Geriatric-Health Promotion and Intelligent Medicine
This project is for midterm test of course "Geriatric Health Promotion and Intelligent Medicine". The aim is to predict ECMO patients about their survival: alive or death as a single target, ignoring brain function. 

This project divides into main stages:
- Preprocess dataset: Analysizing dataset by using bar chart, dealing with the missing value (NaN), classifying the features, and using cross validation to seperate training(80%), validation(10%) and testing(10%) dataset
- Train machine learning methods and deep learning model 
- Compare performance: Accuracy, Sensitivity, Specificity and AUROC

### Download dataset 
The link to download dataset about ECMO patients
``` bash 
https://doi.org/10.1371/journal.pone.0166148.s001
```
### 📦 Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- matplotlib
- numpy

### Install dependencies:

```bash
pip install -r requirements.txt
```
### Preprocess dataset 
```bash
python preprocess_data.py
```
### Train deep learning model (CNN1D)
```bash
python train.py
```
### Test deep learning model (CNN1D)
```bash
python test.py
```
### Train and test machine learning methods
```bash
python Machine_Learning_ALgorithms/decision_tree.py
```
