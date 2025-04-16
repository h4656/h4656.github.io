# Creating a Multi-Class Classifier with Fastai
---
Fastai has developed many libraries that enable the development of artificial intelligence models to be simple and easily accessible. Using the Fast AI course, I developed my own classifier model that can recognise images as being either an airplane, automobile, bird, cat, or dog. The following depicts the methods for creating the classifier model. 

<p align="center">
  
<img src = '/images/multi.png' height = 300>

</p>

The classifier model was developed using the Fast Ai model `is-it-a-bird`, which classifies images as either a bird or a forest.
Get the Jupyter Notebook containing models used in the Fast AI Course by cloning the following Git Repository, then open the 'is-it-a-bird' notebook. 

> 🔗 **Copy:** <a href="https://github.com/lovellbrian/course22.git" target="_blank">Fast AI Course Notebook</a>

```python
git clone https://github.com/lovellbrian/course22.git
```
## Download Images to Create a Dataset
---
To create an image classifier, the model needs a dataset of images labeled with their classification. The Fastai model downloads 200 images of each classification from `Duck Duck Go`. The images are saved to a directory where a folder for each class with the classifier name is created with the associated search images. This is essential to the initial learning of the model as it needs to know the classification of each of the images.  

Additionally, some of the images downloaded may not be suitable to the Fastai Learning model so are removed.

## Train the Model
---
Once the dataset has been collated, the training model can be developed using a `Dataloaders`. A `DataBlock` enables a `Dataloader` object to be easily created from a dataset of sample images for the model. The meaning of the `DataBlock` parameters is depicted in the below table. 


```python
from fastai.vision.all import

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=64)

dls.show_batch(max_n=6)
```



