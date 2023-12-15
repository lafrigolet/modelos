#!/usr/bin/bash
# coding: utf-8

./cnn_test.py -p ./suicide_dataset.small.tensor -o ./suicide_model.small.cnn.pth -s 1000 -w 150 -t 40
./cnn_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnn.5.pth -s 1000 -w 150 -t 40
./cnn_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnn.10.pth -s 1000 -w 150 -t 40
./cnn_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnn.50.pth -s 1000 -w 150 -t 40
./cnn_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnn.100.pth -s 1000 -w 150 -t 40

./cnnlstm_test.py -p ./suicide_dataset.small.tensor -o ./suicide_model.small.cnnlstm.pth -s 1000 -w 150 -t 40
./cnnlstm_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.5.pth -s 1000 -w 150 -t 40
./cnnlstm_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.10.pth -s 1000 -w 150 -t 40
./cnnlstm_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.50.pth -s 1000 -w 150 -t 40
./cnnlstm_test.py -p ./suicide_dataset.tensor -o ./suicide_model.cnnlstm.100.pth -s 1000 -w 150 -t 40


