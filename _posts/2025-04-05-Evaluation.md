# How do I Evaluate the Success of My Model? 
---

The following documents the methods and tools I have used to evaluate the success of the artificial intelligence models I have developed through the Fast Ai course:
- [Confusion Matrices](#confusion)
- [t-SNE](#tsne)

>💡**Note:** Methods of evaluation should be used on the validation set of data and not that used as part of the training set. This ensures that you observe how well the model makes predictions on unseen data. 
<a id='confusion'>

## Confusion Matrices
</a>

Confusion matrices are a common tool used for evaluating classifier models as they provide a qualitative visual for the model's success in making predictions. The confusion matrix has each of the classes along the two axis and compares the predictions made by the model to the true result. 

After creating a Fast Ai Vision Learner Model, a confusion matrix to visualise the learning results can be generated with the code below:

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize = (12,12), dpi = 60)
```
>🔗 Can find my development of a [Mulit-Class Classifier](/posts/2025-04-12-Mulit-Class_Classifier.md) here

An example of a confusion matrix I developed for this Multi-Class Classifier model is depicted below. 
<p align = 'center'>
<img src = '/images/confusion.png'>
</p>
<a id='t-SNE'>
  
## t-SNE
<a id='t-SNE'>
