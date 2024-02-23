
# EmBioMark

This is the repository of the pubblication Knowledge graph embeddings prioritize biomarkers in human diseases.


Here we provide the knowledge graph with whole DisGeNET gene disease associations (GDAs) as well as all the scripts to create knowledge graph embeddngs (KGE) with different models tested in the paper (DistMult, TransE, Graph convolutional network (GCN), Node2Vec, DLemb, BioKG2Vec), to train a classification algorithm (Logistic regression (LR), Support Vector Machine (SVM), Gradient boost descent (XGBOOST)) and to run the predictions of possible GDAs associated to a disease.

Moreover, we provide the best model for GDA that we evaluated during a Grid search cross-validation that we carried out for this work. The model is a SVM trained on the embeddings generated with DLemb algorithm. The GDA representation for this model is the concatenation of the embedding of the disease and the embedding of the gene (figure 1).

![figure 1](./img/pipeline_image.png)


## Environment creation

First of all it is necessary to create a conda environment. Miniconda can be downloaded [here](https://docs.conda.io/en/latest/miniconda.html#installing). After the installation of Miniconda the environment to run this program can be created by running

```
conda env create --file KGE.yml 

```

And then activated by running

```
conda activate KGE
```

## Generate embeddings

To generate the KGE we provide the script located in Scripts/CreateEmbeddings.py. In this script are implemented the aforementioned models. To list the flags of this script run: 

```
./CreateEmbeddings.py -h


Create embeddings given a knowledgegraph

options:
  -h, --help            show this help message and exit
  -k KNOWLEDGEGRAPH, --KnowledgeGraph KNOWLEDGEGRAPH
                        path of the KG in pickle format (networkx Digraph
                        Object)
  -m MODEL, --model MODEL
                        model to use to produce embeddings possible values are:
                        DLemb, BioKG2Vec, DistMult, TransE, GCN, N2V
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train the model for
  -o OUTPUT, --output OUTPUT
                        path in which to save the embeddings
```

To generate the Embeddings using the provided KG and DLemb model trained for 100 epochs run:

```
./CreateEmbeddings.py -k ../KGs/KG.pickle -m DLemb -e 100 -o ../Embeddings/DLembId2Vec.pickle

```

We already provide the embeddings generated with DLemb, BioKG2vec and Node2Vec.


## Train a model

To train a model for predictin GDA we provide the script located in Scripts/TrainModel.py.

To list the parameters of this script run:

```
./TrainModel.py -h


Train and save the model on specific embeddings

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        model type, it can be SVM RF and LR for using
                        respectively support vector machine, random forest or
                        logistic regression
  -p PARAMETERS, --parameters PARAMETERS
                        optional: parameters to be passed to the model, they
                        can be passed in a from of a python dictionary in
                        string from ex : "{'C': 10, 'gamma': 0.1, 'kernel':
                        'rbf'}"
  -f FEATURES, --features FEATURES
                        feature type to use, it can be Sum, Concatenation,
                        Hadmard, Average
  -e EMBEDDINGS, --embeddings EMBEDDINGS
                        path in which the Id2Vec dictionary is
  -o OUTPUT, --output OUTPUT
                        path in which to save the model

```


To train an SVM with parameters = {'C': 10, 'gamma': 0.1, 'kernel':'rbf'} on embeddings generated with DLemb, using concatenation as GDA rappresentation run (this model is already provided):

```
./TrainModel.py -m SVM -p  "{'C': 10, 'gamma': 0.1, 'kernel':'rbf'}" -f Concatenation -e ../Embeddings/DLembId2Vec.pickle -o ../Models/SVM_Concat_DLemb.pickle

``` 


## Predict biomarkers 

We provide the predictions for intervertebral disc degeneration (IDD, CUI:C0158266) in the folder Predictions, to generate the predictions it is sufficient to run the script RunPredictions.py located in Script Folder. The programm accept some flags that can be listed by running:

```
./RunPredictions.py -h


options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path of the trained model for GDA predictions
  -e EMBEDDINGS, --embeddings EMBEDDINGS
                        path in which the Id2Vec dictionary is
  -f FEATURES, --features FEATURES
                        GDA representation type used for predictions, it can be
                        Sum, Concatenation, Hadmard, Average, the model have to
                        be trained with the same representation
  -c CUI, --CUI CUI     cui code for whichrun the predictions
  -o OUTPUT, --output OUTPUT
                        path in which to save the predictions
```

A typical call of the script for generating predictions for IDD and using the pretrained model we provide is as following:


```
./RunPredictions.py -m ../Models/SVM_Concat_DLemb.pickle -e ../Embeddings/DLembId2Vec.pickle -c C0158266 -o ../Predictions/Predictions_C0158266.csv

```

