#!/usr/bin/bash
# coding: utf-8

./cnn_train.py -p ./suicide_dataset.small.tensor -o ./suicide_model.small.cnn.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 5
./cnn_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnn.5.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 5
./cnn_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnn.10.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 10
./cnn_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnn.50.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 50
./cnn_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnn.100.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 100


./cnnlstm_train.py -p ./suicide_dataset.small.tensor -o ./suicide_model.small.cnnlstm.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 5
./cnnlstm_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.5.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 5
./cnnlstm_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.10.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 10
./cnnlstm_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.50.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 50
./cnnlstm_train.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.100.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 100




