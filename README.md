# Relation-Extraction-using-CNN
A Chainer implementation of a Convolutional Network model for relation classification in the SemEval Task 8 dataset. This model performs Multi-Way Classification of Semantic Relations Between Pairs of Nominals in the SemEval 2010 task 8 dataset.

The CNN model is inspired by <a href=http://www.aclweb.org/anthology/C14-1220> Relation Classification via Convolutional Deep Neural Network</a>

Requirements

<b>Requirements</b>
1. Python3
2. Chainer
3. vsmlib
4. numpy
5. Word Embeddings (It can be downloaded from https://nlp.stanford.edu/projects/glove/, the Stanford NLP group has a bunch of open source pre-trained Glove embeddings or you can use your own embeddings. Just specify the path in config.yaml)

<b>Dataset</b>
The Semeval 2010 task 8 dataset can be downloaded from https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview. A small subset of the data has been provided to get you started. The format of the data is as follows:
```
Component-Whole(e2,e1)	12	15	The system as described above has its greatest application in an arrayed configuration of antenna elements .
```
The first part is the label ie, the relation between the nominals present at index 12 and 15 respectively.


<b>Configuration parameters</b>
All the config parameters and the hyperparameters of the model can be specified in the config.yaml file.

<b>Train the model</b>
```
python3 main.py config.yaml
