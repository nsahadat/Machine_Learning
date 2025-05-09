## Data:
Data used from following tabular data:
https://www.kaggle.com/datasets/sohier/large-purchases-by-the-state-of-ca
this data needs to be saved in the ./Data folder of the code

## Code:
### To do:
pip install requirements.txt

### Instructions to follow the codes
data is preprocessed for huggingface dataset using following notebook:
ProcessDataset.ipynb

training GPT2 can be found in the following notebook:
NextValuePrediction_train.ipynb

trained model is saved in the ./gpt2_finetuned directory where model can be loaded and predict the new missing tabular column
NextValuePrediction_Evaluation.ipynb is used to predict the column values

## Results:
final ground truth vs. predicted values can be found in this file:
predicted_value_imputation.csv

## Summary:
1. simple autoencoder by masking column values are tried using this code:
   LargeScaleOrderPrediction.ipynb
2. LLMs tends to work better predicting tabular column values since # of categorical columns more than numerical columns
3. 2 LLMs are tried BERT and GPT2
4. BERT model prediction is very noisy as opposed to GPT2
5. GPT2 resulted more accurate prediction and also less noisy
6. Because of the hardware limitation (12GB GPU NVDIA RTX 3080Ti), I had to use distilgpt2 which requires less memory and memory error can be avoided
