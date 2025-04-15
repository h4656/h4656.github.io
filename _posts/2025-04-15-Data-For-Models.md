# How Much Data Do I Need to Train a Model?

There is a great misconception surrounding the need for a large amount of data required to train a successful artificial intelligence model. 
However, this is not the case. Below discusses methods you should implement in training your model and to best use the data you have available:
1. TOC
{:toc}

## Training and Validation Datasets
While it is important to form the model by learning from the available dataset, it is also essential to have data that can be used to determine how successful your model is – the validation set!
Without a validation set, your model becomes prone to overfitting. This is when the model begins to ‘memorise’ the training data. Whilst the model may be successful on your dataset, there is a high chance that it will result in poor predictions for new data. 
Typically, a split of 80% training and 20% validation is used for the datasets. 

## Using a Pretrained Model
[FastAi Pretrained Models](https://fastai1.fast.ai/vision.models.html)
