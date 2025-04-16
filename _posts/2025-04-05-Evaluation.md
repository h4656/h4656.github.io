# How do I Evaluate the Success of My Model? 
---

The following documents the methods and tools I have used to evaluate the success of the artificial intelligence models I have developed through the Fast Ai course:
- [Confusion Matrices](#confusion)
- [t-SNE](#tsne)

>🔗 Can find my development of a [Mulit-Class Classifier](/posts/2025-04-12-Mulit-Class_Classifier.md) here

>💡**Note:** Methods of evaluation should be used on the validation set of data and not that used as part of the training set. This ensures that you observe how well the model makes predictions on unseen data. 
<a id='confusion'>

## Confusion Matrices 
---
Confusion matrices are a common tool used for evaluating classifier models as they provide a qualitative visual for the model's success in making predictions. The confusion matrix has each of the classes along the two axis and compares the predictions made by the model to the true result. 

After creating a Fast Ai Vision Learner Model, a confusion matrix to visualise the learning results can be generated with the code below:

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize = (12,12), dpi = 60)
```

An example of a confusion matrix I developed for this Multi-Class Classifier model is depicted below. 
<p align = 'center'>
<img src = '/images/confusion.png'>
</p>
<a id='t-SNE'>
  
## t-SNE 
---
T-distributed Stochastic Neighbor Embedding (t-SNE) is a useful tool that enables high dimensional data structures to be visualised in a low dimensional space. Points are modeled by t-SNE as having a neighbouring point in a higher and lower dimension. Using a Gaussian Kernel, the similarity between the two points is calculated and converted into a joint probability. This allows the data which is not linearly structured to be divided by the features identified by the trained model.

```python
from sklearn.manifold import TSNE
import seaborn as sns
hook = hook_output(learn.model[1][-2])
feat = []
classes = []
for x, y in dls.valid:
    with torch.no_grad():
        learn.model.eval()
        _ = learn.model(x)
        feat.append(hook.stored.cpu())
        classes.append(y.cpu())

feat = torch.cat(feat)
classes = torch.cat(classes)
tsne = TSNE(n_components=2, random_state=42)
_tsne = tsne.fit_transform(feat.numpy())
```
The `divergence` can be inspected as per the following. A lower divergence indicates a more successful model as the t-SNE strives to minimize divergence. 
```python
tsne.kl_divergence_
```
The following can be used to develop a scatterplot and provide a visualisation of the t-SNE analysis:
```python
plt.figure(figsize=(6,6))
sns.scatterplot(x=_tsne[:,0], y=_tsne[:,1], hue=[dls.vocab[n] for n in classes])
plt.title('t-SNE Analysis for Multi-Classifier')
plt.show
```
<img src = '/images/tsne.png'>
