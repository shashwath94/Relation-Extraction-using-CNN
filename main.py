
import numpy as np
import yaml
from chainer.dataset import convert
import chainer
from chainer.training import extensions
from preprocess import DataProcessor
from CNN import CNN
import chainer.links as L
import chainer.optimizers as O
from chainer import training
import sys

def main(options):
    
    #load the config params
    gpu = options['gpu']
    data_path = options['path_dataset']
    embeddings_path = options['path_vectors']
    n_epoch = options['epochs']
    batch_size = options['batchsize']
    test = options['test']
    embed_dim = options['embed_dim']
    freeze = options['freeze_embeddings']
    distance_embed_dim = options['distance_embed_dim']

    #load the data
    data_processor = DataProcessor(data_path)
    data_processor.prepare_dataset()
    train_data = data_processor.train_data
    test_data = data_processor.test_data

    vocab = data_processor.vocab
    cnn = CNN(n_vocab=len(vocab), input_channel=1,
                    output_channel=100, 
                    n_label=19, 
                    embed_dim=embed_dim, position_dims=distance_embed_dim, freeze=freeze)
    cnn.load_embeddings(embeddings_path, data_processor.vocab)
    model = L.Classifier(cnn)
    
    #use GPU if flag is set
    if gpu >= 0:
        model.to_gpu()
    
    #setup the optimizer
    optimizer = O.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
    test_iter = chainer.iterators.SerialIterator(test_data, batch_size,repeat=False, shuffle=False) 

    updater = training.StandardUpdater(train_iter, optimizer, converter=convert.concat_examples, device=gpu)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'))

    # Evaluation
    test_model = model.copy()
    test_model.predictor.train = False
    trainer.extend(extensions.Evaluator(test_iter, test_model, device=gpu, converter=convert.concat_examples))


    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=10))


    trainer.run()


if __name__ == '__main__':
    

    options = {}
    if len(sys.argv) > 1:
        path_config = sys.argv[1]
    else:
        print("usage: python3 -m vsmlib.benchmarks.relation_classification <config file>")
        print("config file example can be found at ")
        print("https://github.com/undertherain/vsmlib/blob/master/vsmlib/benchmarks/relation_classification/config.yaml")
        
    with open(path_config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    
    #options["path_data"] = cfg["path_data"]
    options["path_vectors"] = cfg["path_vectors"]
    options["path_dataset"] = cfg["path_dataset"]
    options["gpu"] = cfg["gpu"]
    options["epochs"] = cfg["epochs"]
    options["batchsize"] = cfg["batch_size"]
    options["test"] = cfg["test"]
    options["embed_dim"] = cfg["embed_dim"]
    options["freeze_embeddings"] = cfg["freeze_embedding"]
    options["distance_embed_dim"] = cfg["distance_embed_dim"]

    main(options)