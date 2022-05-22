# Disaster Tweets Classification with BERT PyTorch

The notebook contains code to train BERT base uncased using HuggingFace transformers in PyTorch.<br>
The dataset used is [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data) from Kaggle, where the aim is to predict whether a given tweet is about a real disaster or not. The dataset is slightly imbalanced with 42.9% positive samples.

<br>
The BERT model has F1 score of 81.96% on the validation split (25% of dataset), when trained for 3 epochs with CrossEntropyLoss and AdamW optimizer with learning rate = 3e-5.

## Libraries
* PyTorch
* transformers 
* numpy
* pandas
* sklearn


