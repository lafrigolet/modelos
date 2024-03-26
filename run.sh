#!/bin/bash

#
# To run for every new dataset
#

dataset split --source suicide_dataset/0 --percentage 0.20 --split1 suicide_dataset.$1/test/0 --split2 suicide_dataset.$1/train/0 && \
dataset split --source suicide_dataset/1 --percentage 0.20 --split1 suicide_dataset.$1/test/1 --split2 suicide_dataset.$1/train/1 && \
dataset crop --source suicide_dataset.$1/test/0 --destination suicide_dataset.$1/cropped/test/0 --width 150 --height 40 && \
dataset crop --source suicide_dataset.$1/test/1 --destination suicide_dataset.$1/cropped/test/1 --width 150 --height 40 && \
dataset crop --source suicide_dataset.$1/train/0 --destination suicide_dataset.$1/cropped/train/0 --width 150 --height 40 && \
dataset crop --source suicide_dataset.$1/train/1 --destination suicide_dataset.$1/cropped/train/1 --width 150 --height 40 && \
dataset filter --source suicide_dataset.$1/cropped/test/0 --destination suicide_dataset.$1/filterhw/test/0 --mhm machinehand_model.pth && \
dataset filter --source suicide_dataset.$1/cropped/test/1 --destination suicide_dataset.$1/filterhw/test/1 --mhm machinehand_model.pth && \
dataset filter --source suicide_dataset.$1/cropped/train/0 --destination suicide_dataset.$1/filterhw/train/0 --mhm machinehand_model.pth && \
dataset filter --source suicide_dataset.$1/cropped/train/1 --destination suicide_dataset.$1/filterhw/train/1 --mhm machinehand_model.pth && \
dataset balanceup --dir1 suicide_dataset.$1/filterhw/test/0 --dir2 suicide_dataset.$1/filterhw/test/1 && \
dataset balanceup --dir1 suicide_dataset.$1/filterhw/train/0 --dir2 suicide_dataset.$1/filterhw/train/1 && \ 
dataset tensorize --source suicide_dataset.$1/filterhw/test/0 --destination suicide_dataset.$1/tensors/test/0 && \
dataset tensorize --source suicide_dataset.$1/filterhw/test/1 --destination suicide_dataset.$1/tensors/test/1 && \
dataset tensorize --source suicide_dataset.$1/filterhw/train/0 --destination suicide_dataset.$1/tensors/train/0 && \
dataset tensorize --source suicide_dataset.$1/filterhw/train/1 --destination suicide_dataset.$1/tensors/train/1 && \
train.py --model cnn -p ./suicide_dataset.$1/tensors -o ./suicide_model.cnn.$1.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 10 && \
train.py --model cnnlstm -p ./suicide_dataset.$1/tensors -o ./suicide_model.cnnlstm.$1.pth -b 64 -s 1000 -e 70 -w 150 -t 40 -l 0.0001 -n 10 && \
test.py --model cnn --pth ./suicide_model.cnn.$1.pth --csv ./suicide_model.cnn.$1.csv --png ./suicide_model.cnn.$1.png -p ./suicide_dataset.$1/tensors/test  -s 1000 -w 150 -t 40 && \
test.py --model cnnlstm --pth ./suicide_model.cnnlstm.$1.pth --csv ./suicide_model.cnnlstm.$1.csv --png ./suicide_model.cnnlstm.$1.png -p ./suicide_dataset.$1/tensors/test  -s 1000 -w 150 -t 40 
