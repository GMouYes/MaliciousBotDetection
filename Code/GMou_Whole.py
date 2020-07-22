#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:04:43 2020

@author: G.Mou
"""

# torch packages
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

# python's own
import time
import numpy as np
import itertools
import pickle
import math
import gc
import faulthandler

# scikit-learn
from sklearn.metrics import classification_report
# I wrote the following
from model import save_model, load_model, get_model_setting
from data import myDataset, readData
# extra libraries
import matplotlib.pyplot as plt
from pprint import pprint

GLOBAL_BEST_ACC = 0.0

class Trainer(object):
    """Trainer."""
    def __init__(self, trainer_args, args_dict, gridSearch):
        self.n_epochs = args_dict["epochs"]
        self.batch_size = args_dict["batch_size"]
        self.validate = trainer_args["validate"]
        self.save_best_dev = trainer_args["save_best_dev"]
        self.use_cuda = trainer_args["use_cuda"]
        self.print_every_step = trainer_args["print_every_step"]
        self.optimizer = trainer_args["optimizer"]
        self.model_path = args_dict["model_path"]

        # set for saving the best grid search results
        self.grid = {**gridSearch, **args_dict}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_cuda else "cpu")

        self.training_size = args_dict["training_size"]
        self.valid_size = args_dict["valid_size"]

    def train(self, network, train_data, dev_data=None):
        # transfer model to gpu if available
        network = network.to(self.device)

        train_loss, validate_loss = [], []
        # define Tester over dev data
        if self.validate:
            default_valid_args = {
                "batch_size": self.batch_size,
                "use_cuda": self.use_cuda,
                "valid_size": self.valid_size,
                }
            validator = Tester(**default_valid_args)
        
        for epoch in range(1, self.n_epochs + 1):
            # turn on network training mode
            network.train()

            # one forward and backward pass
            epoch_train_loss = self._train_step(train_data, network, n_print=self.print_every_step, epoch=epoch)
            train_loss.append(epoch_train_loss)
            print('epoch {}:'.format(epoch), end=" ")

            # validation
            if self.validate:
                if dev_data is None:
                    raise RuntimeError(
                        "Self.validate is True in trainer, "
                        "but dev_data is None."
                        "Please provide the validation data.")
                test_acc, epoch_valid_loss = validator.test(network, dev_data)
                validate_loss.append(epoch_valid_loss)
                if self.save_best_dev and self.best_eval_result(test_acc):
                    save_model(network, self.model_path, self.grid)
                    print("Saved better model selected by validation.")
            gc.collect()

        self.plot_loss(train_loss, validate_loss)


    def plot_loss(self, train_loss, valid_loss):
        
        fig = plt.figure()
        ax = plt.subplot(221)
        ax.set_title('train loss')
        ax.plot(train_loss,'r-')

        ax = plt.subplot(222)
        ax.set_title('validation loss') 
        ax.plot(valid_loss,'b.')

        plt.savefig('trainLoss_vs_validLoss_{}.pdf'.format(self.grid['learning_rate']))
        plt.close()


    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.
        """
        loss = 0
        loss_record = 0.0
        for i, data in enumerate(data_iterator):
            data = [item.to(self.device) for item in data]

            label = data[-1]
            data = data[:-1]

            self.optimizer.zero_grad()
            logit = network(data)

            criterion = torch.nn.BCEWithLogitsLoss()

            loss = criterion(logit, label)
            loss.backward()
            self.optimizer.step()

            loss_record += loss.item()

        return loss_record

    def best_eval_result(self, test_acc):       
        """Check if the current epoch yields better validation results.
        """
        global GLOBAL_BEST_ACC
        
        if test_acc > GLOBAL_BEST_ACC:
            GLOBAL_BEST_ACC = test_acc
            return True
        return False
        
class Tester(object):
    """Tester."""

    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.use_cuda = kwargs["use_cuda"]
        self.testing_size = kwargs["valid_size"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_cuda else "cpu")

    def test(self, network, dev_data):
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()

        mean_acc, valid_loss = 0.0, 0.0
        pred_list, truth_list = np.empty((0,1)), np.empty((0,1))
        
        for i, data in enumerate(dev_data):
            data = [item.to(self.device) for item in data]

            label = data[-1]
            data = data[:-1]

            with torch.no_grad():
                outputs = network(data)

                criterion = torch.nn.BCEWithLogitsLoss()
                valid_loss+=criterion(outputs, label).item()
                
            _prediction = outputs.cpu().numpy()
            pred_list = np.concatenate((pred_list,_prediction), axis=0)

            _truth = label.cpu().numpy()
            truth_list = np.concatenate((truth_list,_truth), axis=0)

        acc = generate_acc(pred_list,truth_list,need_sigmoid=True)
        print("[Tester] Accuracy: {:.4f}".format(acc))

        return acc, valid_loss


class Predictor(object):
    """Predictor."""

    def __init__(self, testing_size, batch_size=128, use_cuda=True):
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.testing_size = testing_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_cuda else "cpu")

    def predict(self, network, test_data):
        network = network.to(self.device)

        # turn on the testing mode; clean up the history
        network.eval()

        truth_list,pred_list,mask_list = np.empty((0,1)),np.empty((0,1)),np.empty((0,1))

        acc = 0.0

        for i, data in enumerate(test_data):
            data = [item.to(self.device) for item in data]

            label = data[-1]
            data = data[:-1]

            with torch.no_grad():
                outputs = network(data)

            _prediction = outputs.cpu().numpy()
            pred_list = np.concatenate((pred_list,_prediction), axis=0)

            _truth = label.cpu().numpy()
            truth_list = np.concatenate((truth_list,_truth), axis=0)

        return pred_list,truth_list

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def generate_acc(pred, truth, need_sigmoid=True):
    if(need_sigmoid):
        sigmoid_v = np.vectorize(sigmoid)
        pred = sigmoid_v(pred)
    pred = pred > .5
    acc = np.sum(pred==truth) / len(pred)
    return acc

def train(args_dict, gridSearch, data):
    """Train model.
    """
    print("Training...")
    # load data
    data_train = data["data_train"]
    # define model
    args = {**args_dict, **gridSearch}
    model = get_model_setting(**args)
    # define trainer
    trainer_args = {
        "validate": True,
        "save_best_dev": True,
        "use_cuda": True,
        "print_every_step": 1000,
        "optimizer": torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=0.1*args['learning_rate']),
    }
    trainer = Trainer(trainer_args, args_dict, gridSearch)
    
    # train
    data_val = data["data_valid"]
    trainer.train(model, data_train, dev_data=data_val)

    print('')

def infer(data):
    """Inference using model.
    """
    print("Predicting...")
    # define model
    with open('hyper.pkl','rb') as f:
        final_grid = pickle.load(f)
    model = get_model_setting(**final_grid)
    load_model(model, final_grid['model_path'])

    # define predictor
    predictor = Predictor(batch_size=final_grid["batch_size"], use_cuda=True,testing_size=data["testing_size"])

    # predict
    data_test = data["data_test"]
    y_pred,y_true = predictor.predict(model, data_test)

    sigmoid_v = np.vectorize(sigmoid)
    y_pred = sigmoid_v(y_pred)

    np.save("pred_prob.npy", y_pred)

    y_pred = y_pred>.5

    np.save('prediction.npy',y_pred)
    np.save('ground_truth.npy',y_true)

    labels = [0,1]
    target_names = ["human", "bot"]
    clf = classification_report(y_true,y_pred, labels=labels, target_names=target_names, digits=4)
    print(clf)
    
    return

def loadData(datatype:str, shuffle=False, batch_size=128, num_workers=8):
    # get dataset
    X, y = readData(datatype)
    data_set = myDataset(X,y)
    data_size = len(data_set.y)
    print("{} size: {}".format(datatype, data_size))
    data_loader = DataLoader(data_set, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers
                            )
    return data_loader, data_size

def pre():
    """Pre-process model."""
    print("Pre-processing...")

    # things to be fixed
    args_dict = {
        "batch_size":   128,
        "epochs":       50,
        "iptDim":       256,
        "gafmtfDim":    256,
        "wordDim":      768,
        "wordLen":      80,
        "sentDim":      512,
        "sentLen":      80,
        "tradDim":      84,
        "output_size":  1,
        "model_path":   "model.pkl",
        }

    # things to be tuned
    gridSearch = {
        "cnn_out":          [128],  # <768
        "lstm_out":         [256],  # <768
        "lstm_dropout":     [0.2],  # <0.5
        "tbHeads":          [1024],  # <1024
        "textHeads":        [256],  # <1024
        "layer1_size":      [512],  # <1024
        "layer1_dropout":   [0.05],  # <0.5
        "layer2_size":      [128],
        "layer2_dropout":   [0.05],
        "learning_rate":    [8e-4], # 1e-3~1e-7
        }

    return args_dict, gridSearch

def search(args_dict, gridSearch, data):
    for grid in [dict(zip(gridSearch.keys(),v)) for v in itertools.product(*gridSearch.values())]:
        if grid["layer1_size"] <= grid["layer2_size"]:
            continue
        train(args_dict, grid, data)
        gc.collect()
    return True

def wrapUp(task:str, func, **args):
    start = time.time()
    returnObject = func(**args)
    end = time.time()
    period = end-start
    print("{}:".format(task), end="")
    print("It took {} hour {} min {} sec".format(period//3600,(period%3600)//60,int(period%60)))
    print("")
    gc.collect()
    return returnObject

def main():
    # setting up seeds
    seed = 1
    # manually fix random seq
    np.random.seed(seed)
    torch.manual_seed(seed) 
    
    # preprocessing
    args_dict,gridSearch = wrapUp(task="Preprocessing", func=pre)
    
    # load training & validation
    data_train,training_size = wrapUp(task="load train", func=loadData, datatype="train", shuffle=True)
    data_valid,valid_size = wrapUp(task="load valid", func=loadData, datatype="valid")

    args_dict["training_size"] = training_size
    args_dict["valid_size"] = valid_size
    data = {"data_train":data_train, "data_valid":data_valid}
    
    # training
    Status = wrapUp(task="Training", func=search, args_dict=args_dict, gridSearch=gridSearch, data=data)

    # load testing
    data_test,testing_size = wrapUp(task="load test", func=loadData, datatype="test")
    data = {
        "data_test": data_test,
        "testing_size": testing_size,
        "model_path": args_dict["model_path"],
    }
    # testing
    Status = wrapUp(task="Predicting", func=infer, data=data)

    return True

# code starts here
if __name__ == "__main__":
    faulthandler.enable()
    main()
    faulthandler.disable()
    
