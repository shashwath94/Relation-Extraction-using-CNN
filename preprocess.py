import numpy as np
import os
import sys
import vsmlib
from collections import defaultdict




class DataProcessor(object):
    def __init__(self, data_path):
        self.train_data_path = os.path.join(data_path, "train.txt")
        self.test_data_path = os.path.join(data_path, "test.txt")
        self.file_list = [self.train_data_path, self.test_data_path]
        self.max_sent_length = max(self.get_maxlen(self.file_list))
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.vocab["<pad>"]


    def prepare_dataset(self):
        # load train/test data
        sys.stderr.write("loading dataset...")
        self.train_data = self.load_dataset("train")
        self.test_data = self.load_dataset("test")
        
        sys.stderr.write("done.\n")


    def load_dataset(self, _type):
        
        if _type == "train":
            path = self.train_data_path
        else:
            path = self.test_data_path
            
        dataset = self.create_input(path, self.max_sent_length)
        return dataset

    def get_maxlen(self, files):
        
        maxSentenceLen = [0, 0]
        for fileIdx in range(len(files)):
            file = files[fileIdx]
            for line in open(file):
                splits = line.strip().split('\t')

                label = splits[0]


                sentence = splits[3]
                tokens = sentence.split(" ")
                maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
        return maxSentenceLen
    
    
    def create_input(self, path, maxSentenceLen=100):     
        """Build the input matrices for tokens and distances"""
        
        dataset = []
        labels = []
        positionMatrix1 = []
        positionMatrix2 = []
        tokenMatrix = []

        labelsMapping = {'Other':0,
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2,
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4,
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6,
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}

        words = {}
        


        distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
        
        minDistance = -30
        maxDistance = 30
        for dis in range(minDistance,maxDistance+1):
            distanceMapping[dis] = len(distanceMapping)
        #sprint(distanceMapping)

        for line in open(path):
            split = line.strip().split('\t')

            label = split[0]
            pos1 = split[1]
            pos2 = split[2]
            sentence = split[3]
            tokens = sentence.split(" ")


            tokenIds = np.zeros(maxSentenceLen, dtype='int32')
            positionValues1 = np.zeros(maxSentenceLen, dtype='int32')
            positionValues2 = np.zeros(maxSentenceLen, dtype='int32')

            for idx in range(0, min(maxSentenceLen, len(tokens))):
                
                tokenIds[idx] = int(self.vocab[tokens[idx]])
                
                distance1 = idx - int(pos1)
                distance2 = idx - int(pos2)
                
                if distance1 in distanceMapping:
                    
                    positionValues1[idx] = distanceMapping[distance1]
                elif distance1 <= minDistance:
                    positionValues1[idx] = distanceMapping['LowerMin']
                else:
                    positionValues1[idx] = distanceMapping['GreaterMax']

                if distance2 in distanceMapping:
                    positionValues2[idx] = distanceMapping[distance2]
                elif distance2 <= minDistance:
                    positionValues2[idx] = distanceMapping['LowerMin']
                else:
                    positionValues2[idx] = distanceMapping['GreaterMax']

            
            
            dataset.append((tokenIds, positionValues1, positionValues2, np.array(labelsMapping[label], dtype='int32')))
           
        return dataset

