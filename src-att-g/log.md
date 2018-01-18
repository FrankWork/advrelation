$ conv + max pool 0.7909

==============================================================================
INFO:tensorflow:CNNModel/pos1_embed
INFO:tensorflow:CNNModel/pos2_embed
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/kernel
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/bias
INFO:tensorflow:CNNModel/semeval_graph/dense/kernel
INFO:tensorflow:CNNModel/semeval_graph/dense/bias

Epoch 0 sem 2.81 0.19 0.2942 time 7.20
Epoch 1 sem 2.31 0.34 0.4866 time 2.88
Epoch 2 sem 1.90 0.45 0.5794 time 2.56
Epoch 3 sem 1.61 0.54 0.6443 time 2.55
Epoch 4 sem 1.44 0.59 0.6810 time 2.36
Epoch 5 sem 1.32 0.63 0.7028 time 2.42
Epoch 6 sem 1.23 0.66 0.7195 time 2.57
Epoch 7 sem 1.16 0.68 0.7310 time 2.62
Epoch 8 sem 1.11 0.69 0.7403 time 2.34
Epoch 9 sem 1.05 0.71 0.7467 time 2.52
Epoch 10 sem 1.03 0.72 0.7495 time 2.61
Epoch 11 sem 1.00 0.73 0.7531 time 2.67
Epoch 12 sem 0.96 0.74 0.7610 time 2.85
Epoch 13 sem 0.93 0.75 0.7592 time 2.42
Epoch 14 sem 0.91 0.75 0.7667 time 1.97
Epoch 15 sem 0.87 0.76 0.7678 time 2.53
Epoch 16 sem 0.87 0.76 0.7685 time 2.67
Epoch 17 sem 0.84 0.78 0.7727 time 2.54
Epoch 18 sem 0.83 0.78 0.7753 time 2.54
Epoch 19 sem 0.80 0.79 0.7753 time 2.54
Epoch 20 sem 0.80 0.78 0.7785 time 2.04
Epoch 21 sem 0.78 0.80 0.7778 time 2.74
Epoch 22 sem 0.77 0.80 0.7767 time 2.01
Epoch 23 sem 0.76 0.80 0.7760 time 1.99
Epoch 24 sem 0.73 0.81 0.7745 time 2.00
Epoch 25 sem 0.74 0.81 0.7799 time 1.99
Epoch 26 sem 0.72 0.81 0.7785 time 2.52
Epoch 27 sem 0.71 0.82 0.7788 time 1.96
Epoch 28 sem 0.71 0.82 0.7799 time 1.79
Epoch 29 sem 0.69 0.83 0.7799 time 1.77
Epoch 30 sem 0.68 0.83 0.7845 time 2.04
Epoch 31 sem 0.68 0.83 0.7806 time 2.52
Epoch 32 sem 0.67 0.83 0.7824 time 2.00
Epoch 33 sem 0.67 0.83 0.7835 time 1.97
Epoch 34 sem 0.65 0.84 0.7788 time 1.97
Epoch 35 sem 0.64 0.84 0.7789 time 1.98
Epoch 36 sem 0.62 0.85 0.7806 time 1.96
Epoch 37 sem 0.62 0.85 0.7806 time 2.04
Epoch 38 sem 0.62 0.84 0.7849 time 2.08
Epoch 39 sem 0.61 0.85 0.7860 time 2.63
Epoch 40 sem 0.62 0.84 0.7860 time 2.52
Epoch 41 sem 0.60 0.85 0.7838 time 1.81
Epoch 42 sem 0.60 0.85 0.7838 time 2.02
Epoch 43 sem 0.58 0.86 0.7838 time 1.65
Epoch 44 sem 0.58 0.86 0.7849 time 1.47
Epoch 45 sem 0.59 0.86 0.7888 time 1.60
Epoch 46 sem 0.57 0.86 0.7820 time 1.76
Epoch 47 sem 0.58 0.86 0.7849 time 1.63
Epoch 48 sem 0.58 0.86 0.7870 time 1.60
Epoch 49 sem 0.56 0.87 0.7909 time 1.58
Done training, best_epoch: 49, best_acc: 0.7909
duration: 0.03 hours



$ cnn + att

INFO:tensorflow:CNNModel/pos1_embed
INFO:tensorflow:CNNModel/pos2_embed
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/kernel
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/bias
INFO:tensorflow:CNNModel/semeval_graph/dense/kernel
INFO:tensorflow:CNNModel/semeval_graph/dense/bias

INFO:tensorflow:CNNModel/semeval_graph/att1-0/dense/kernel
INFO:tensorflow:CNNModel/semeval_graph/att1-0/dense/bias
INFO:tensorflow:CNNModel/semeval_graph/att1-0/dense_1/kernel
INFO:tensorflow:CNNModel/semeval_graph/att1-0/dense_1/bias
INFO:tensorflow:CNNModel/semeval_graph/att1-0/dense_2/kernel
INFO:tensorflow:CNNModel/semeval_graph/att1-0/dense_2/bias
INFO:tensorflow:CNNModel/semeval_graph/att2-0/dense/kernel
INFO:tensorflow:CNNModel/semeval_graph/att2-0/dense/bias
INFO:tensorflow:CNNModel/semeval_graph/att2-0/dense_1/kernel
INFO:tensorflow:CNNModel/semeval_graph/att2-0/dense_1/bias
INFO:tensorflow:CNNModel/semeval_graph/att2-0/dense_2/kernel
INFO:tensorflow:CNNModel/semeval_graph/att2-0/dense_2/bias

INFO:tensorflow:Train/CNNModel/semeval_graph/att2-0/ln/Variable
INFO:tensorflow:Train/CNNModel/semeval_graph/att2-0/ln/Variable_1
INFO:tensorflow:Train/CNNModel/semeval_graph/att1-0/ln/Variable
INFO:tensorflow:Train/CNNModel/semeval_graph/att1-0/ln/Variable_1

INFO:tensorflow:Valid/CNNModel/semeval_graph/att1-0/ln/Variable
INFO:tensorflow:Valid/CNNModel/semeval_graph/att1-0/ln/Variable_1
INFO:tensorflow:Valid/CNNModel/semeval_graph/att2-0/ln/Variable
INFO:tensorflow:Valid/CNNModel/semeval_graph/att2-0/ln/Variable_1
