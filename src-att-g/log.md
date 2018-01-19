$ conv + max pool (reduce max) 0.7909

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



$ cnn + mh att

================================================================================
INFO:tensorflow:CNNModel/pos1_embed
INFO:tensorflow:CNNModel/pos2_embed
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/kernel
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/bias
INFO:tensorflow:CNNModel/semeval_graph/att1/q/kernel
INFO:tensorflow:CNNModel/semeval_graph/att1/k/kernel
INFO:tensorflow:CNNModel/semeval_graph/att1/v/kernel
INFO:tensorflow:CNNModel/semeval_graph/att1/output_transform/kernel
INFO:tensorflow:CNNModel/semeval_graph/att2/q/kernel
INFO:tensorflow:CNNModel/semeval_graph/att2/k/kernel
INFO:tensorflow:CNNModel/semeval_graph/att2/v/kernel
INFO:tensorflow:CNNModel/semeval_graph/att2/output_transform/kernel
INFO:tensorflow:CNNModel/semeval_graph/dense/kernel
INFO:tensorflow:CNNModel/semeval_graph/dense/bias

Epoch 0 sem 2.49 0.27 0.5205 time 10.10
Epoch 1 sem 1.50 0.56 0.6595 time 8.33
Epoch 2 sem 1.23 0.63 0.6988 time 7.86
Epoch 3 sem 1.06 0.68 0.7217 time 7.87
Epoch 4 sem 0.94 0.71 0.7438 time 7.94
Epoch 5 sem 0.86 0.74 0.7556 time 8.07
Epoch 6 sem 0.81 0.75 0.7460 time 7.98
Epoch 7 sem 0.71 0.78 0.7681 time 7.58
Epoch 8 sem 0.67 0.79 0.7656 time 8.14
Epoch 9 sem 0.62 0.81 0.7749 time 7.70
Epoch 10 sem 0.60 0.81 0.7695 time 8.16
Epoch 11 sem 0.56 0.83 0.7660 time 7.64
Epoch 12 sem 0.52 0.84 0.7757 time 7.72
Epoch 13 sem 0.50 0.84 0.7756 time 8.06
Epoch 14 sem 0.45 0.86 0.7735 time 7.65
Epoch 15 sem 0.45 0.86 0.7685 time 7.73
Epoch 16 sem 0.41 0.88 0.7728 time 7.76
Epoch 17 sem 0.40 0.88 0.7714 time 7.57
Epoch 18 sem 0.36 0.89 0.7732 time 7.74
Epoch 19 sem 0.34 0.90 0.7749 time 7.70
Epoch 20 sem 0.34 0.90 0.7749 time 7.72
Epoch 21 sem 0.32 0.91 0.7824 time 7.79
Epoch 22 sem 0.30 0.92 0.7806 time 7.94
Epoch 23 sem 0.29 0.92 0.7721 time 7.67
Epoch 24 sem 0.27 0.92 0.7721 time 7.70
Epoch 25 sem 0.25 0.93 0.7785 time 7.70
Epoch 26 sem 0.26 0.93 0.7785 time 7.76
Epoch 27 sem 0.24 0.93 0.7828 time 7.64
Epoch 28 sem 0.24 0.93 0.7774 time 8.00
Epoch 29 sem 0.22 0.94 0.7736 time 7.68


$ cnn + dot att


Epoch 0 sem 2.51 0.28 0.4906 time 9.17
Epoch 1 sem 1.67 0.52 0.6307 time 7.39
Epoch 2 sem 1.40 0.60 0.6649 time 6.89
Epoch 3 sem 1.28 0.63 0.6881 time 6.96
Epoch 4 sem 1.19 0.66 0.6970 time 6.84
Epoch 5 sem 1.15 0.67 0.7113 time 6.92
Epoch 6 sem 1.10 0.69 0.7056 time 6.98
Epoch 7 sem 1.06 0.69 0.7085 time 6.75
Epoch 8 sem 1.05 0.70 0.7203 time 6.68
Epoch 9 sem 1.02 0.71 0.7170 time 7.01
Epoch 10 sem 1.00 0.71 0.7203 time 6.67
Epoch 11 sem 0.99 0.71 0.7188 time 6.92
Epoch 12 sem 0.96 0.72 0.7210 time 6.72
Epoch 13 sem 0.95 0.73 0.7228 time 7.62
Epoch 14 sem 0.93 0.74 0.7295 time 7.15
Epoch 15 sem 0.91 0.74 0.7274 time 6.94
Epoch 16 sem 0.91 0.74 0.7270 time 6.70
Epoch 17 sem 0.89 0.74 0.7281 time 6.79
Epoch 18 sem 0.87 0.75 0.7299 time 6.79
Epoch 19 sem 0.86 0.75 0.7295 time 7.11

$ cnn + dot att + max pool

================================================================================
INFO:tensorflow:CNNModel/pos1_embed
INFO:tensorflow:CNNModel/pos2_embed
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/kernel
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/bias
INFO:tensorflow:CNNModel/semeval_graph/dense/kernel
INFO:tensorflow:CNNModel/semeval_graph/dense/bias
Epoch 0 sem 2.70 0.22 0.3345 time 13.56
Epoch 1 sem 1.91 0.44 0.5876 time 11.02
Epoch 2 sem 1.47 0.58 0.6735 time 7.08
Epoch 3 sem 1.27 0.64 0.6971 time 6.93
Epoch 4 sem 1.18 0.67 0.7120 time 7.09
Epoch 5 sem 1.09 0.69 0.7224 time 6.95
Epoch 6 sem 1.04 0.70 0.7206 time 6.88
Epoch 7 sem 1.01 0.71 0.7235 time 6.98
Epoch 8 sem 0.98 0.72 0.7331 time 7.37
Epoch 9 sem 0.95 0.73 0.7431 time 7.51
Epoch 10 sem 0.92 0.74 0.7385 time 8.09
Epoch 11 sem 0.88 0.75 0.7442 time 7.70
Epoch 12 sem 0.86 0.76 0.7474 time 7.99
Epoch 13 sem 0.85 0.76 0.7481 time 15.40
Epoch 14 sem 0.82 0.77 0.7520 time 16.20
Epoch 15 sem 0.80 0.78 0.7567 time 17.22
Epoch 16 sem 0.80 0.78 0.7545 time 12.25
Epoch 17 sem 0.78 0.78 0.7620 time 10.90
Epoch 18 sem 0.76 0.79 0.7603 time 11.50
Epoch 19 sem 0.74 0.80 0.7638 time 10.19
Epoch 20 sem 0.74 0.80 0.7681 time 10.83
Epoch 21 sem 0.72 0.81 0.7710 time 12.56
Epoch 22 sem 0.70 0.81 0.7728 time 12.29
Epoch 23 sem 0.69 0.81 0.7685 time 10.91
Epoch 24 sem 0.69 0.82 0.7699 time 11.04
Epoch 25 sem 0.66 0.83 0.7638 time 10.73
Epoch 26 sem 0.67 0.82 0.7724 time 10.24
Epoch 27 sem 0.66 0.82 0.7728 time 9.92
Epoch 28 sem 0.65 0.83 0.7695 time 11.19
Epoch 29 sem 0.63 0.84 0.7778 time 9.85
Epoch 30 sem 0.64 0.83 0.7710 time 12.73
Epoch 31 sem 0.63 0.83 0.7695 time 11.32
Epoch 32 sem 0.61 0.84 0.7706 time 11.71
Epoch 33 sem 0.61 0.84 0.7660 time 9.96
Epoch 34 sem 0.59 0.85 0.7781 time 10.58
Epoch 35 sem 0.59 0.84 0.7760 time 11.27
Epoch 36 sem 0.60 0.84 0.7703 time 10.22
Epoch 37 sem 0.58 0.85 0.7667 time 9.98
Epoch 38 sem 0.58 0.85 0.7695 time 10.29
Epoch 39 sem 0.56 0.85 0.7713 time 10.57
Epoch 40 sem 0.57 0.85 0.7745 time 9.76
Epoch 41 sem 0.55 0.86 0.7759 time 10.15
Epoch 42 sem 0.55 0.86 0.7767 time 9.87
Epoch 43 sem 0.54 0.87 0.7731 time 9.89
Epoch 44 sem 0.53 0.86 0.7713 time 9.86
Epoch 45 sem 0.54 0.87 0.7816 time 9.64
Epoch 46 sem 0.53 0.87 0.7759 time 10.66
Epoch 47 sem 0.52 0.87 0.7720 time 10.36
Epoch 48 sem 0.52 0.87 0.7745 time 10.91
Epoch 49 sem 0.52 0.87 0.7706 time 9.53
Done training, best_epoch: 45, best_acc: 0.7816
duration: 0.14 hours

$ cnn + tanh att + max pooling

Epoch 0 sem 2.76 0.19 0.3123 time 15.12
Epoch 1 sem 2.27 0.33 0.4872 time 13.77
Epoch 2 sem 1.80 0.47 0.6061 time 13.10
Epoch 3 sem 1.50 0.57 0.6664 time 13.28
Epoch 4 sem 1.34 0.61 0.6978 time 13.59
Epoch 5 sem 1.24 0.64 0.7135 time 14.88
Epoch 6 sem 1.16 0.67 0.7328 time 14.56
Epoch 7 sem 1.08 0.69 0.7378 time 13.21
Epoch 8 sem 1.04 0.71 0.7478 time 12.72
Epoch 9 sem 1.00 0.72 0.7513 time 15.99
Epoch 10 sem 0.98 0.72 0.7574 time 12.92
Epoch 11 sem 0.93 0.74 0.7631 time 13.84
Epoch 12 sem 0.91 0.75 0.7663 time 14.23
Epoch 13 sem 0.87 0.76 0.7688 time 12.90
Epoch 14 sem 0.86 0.76 0.7724 time 15.38
Epoch 15 sem 0.84 0.77 0.7770 time 13.98
Epoch 16 sem 0.81 0.78 0.7745 time 13.54
Epoch 17 sem 0.81 0.78 0.7763 time 12.34
Epoch 18 sem 0.79 0.79 0.7810 time 12.08
Epoch 19 sem 0.75 0.80 0.7870 time 15.16
Epoch 20 sem 0.75 0.80 0.7835 time 13.88
Epoch 21 sem 0.76 0.79 0.7856 time 12.02
Epoch 22 sem 0.74 0.80 0.7842 time 13.00
Epoch 23 sem 0.72 0.81 0.7863 time 12.66
Epoch 24 sem 0.72 0.81 0.7828 time 12.98
Epoch 25 sem 0.70 0.82 0.7885 time 12.52
Epoch 26 sem 0.69 0.82 0.7895 time 15.06
Epoch 27 sem 0.68 0.82 0.7906 time 13.31
Epoch 28 sem 0.66 0.83 0.7885 time 16.04
Epoch 29 sem 0.66 0.83 0.7888 time 12.61
Epoch 30 sem 0.66 0.84 0.7888 time 12.75
Epoch 31 sem 0.65 0.83 0.7885 time 12.91
Epoch 32 sem 0.64 0.84 0.7920 time 12.35
Epoch 33 sem 0.64 0.84 0.7892 time 12.63
Epoch 34 sem 0.61 0.85 0.7910 time 11.61
Epoch 35 sem 0.61 0.85 0.7928 time 12.95
Epoch 36 sem 0.62 0.85 0.7935 time 13.06
Epoch 37 sem 0.60 0.85 0.7917 time 9.14
Epoch 38 sem 0.60 0.85 0.7978 time 8.61
Epoch 39 sem 0.59 0.86 0.7924 time 9.21
Epoch 40 sem 0.59 0.85 0.7885 time 8.79
Epoch 41 sem 0.58 0.86 0.7928 time 8.77
Epoch 42 sem 0.57 0.86 0.7903 time 8.63
Epoch 43 sem 0.57 0.86 0.7974 time 8.86
Epoch 44 sem 0.56 0.86 0.7985 time 8.76
Epoch 45 sem 0.56 0.86 0.7963 time 9.16
Epoch 46 sem 0.56 0.86 0.7974 time 8.81
Epoch 47 sem 0.55 0.86 0.7988 time 8.72
Epoch 48 sem 0.56 0.86 0.7956 time 9.13
Epoch 49 sem 0.54 0.87 0.7974 time 8.80
Done training, best_epoch: 47, best_acc: 0.7988
duration: 0.17 hours

0.7941

two attention: 0.7935
att + proj:    0.7977


$ mh attention + cnn + max pooling

================================================================================
INFO:tensorflow:CNNModel/pos1_embed
INFO:tensorflow:CNNModel/pos2_embed
INFO:tensorflow:CNNModel/semeval_graph/multihead_attention/q/kernel
INFO:tensorflow:CNNModel/semeval_graph/multihead_attention/k/kernel
INFO:tensorflow:CNNModel/semeval_graph/multihead_attention/v/kernel
INFO:tensorflow:CNNModel/semeval_graph/multihead_attention/output_transform/kernel
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/kernel
INFO:tensorflow:CNNModel/semeval_graph/conv_block1/conv-3/bias
INFO:tensorflow:CNNModel/semeval_graph/dense/kernel
INFO:tensorflow:CNNModel/semeval_graph/dense/bias
Epoch 0 sem 2.55 0.25 0.3647 time 10.54
Epoch 1 sem 1.90 0.41 0.5472 time 10.60
Epoch 2 sem 1.49 0.54 0.6520 time 10.65
Epoch 3 sem 1.26 0.61 0.6984 time 9.57
Epoch 4 sem 1.13 0.66 0.7228 time 8.83
Epoch 5 sem 1.05 0.68 0.7459 time 7.77
Epoch 6 sem 0.95 0.71 0.7545 time 9.82
Epoch 7 sem 0.88 0.72 0.7716 time 9.95
Epoch 8 sem 0.85 0.73 0.7741 time 9.33
Epoch 9 sem 0.81 0.75 0.7684 time 9.73
Epoch 10 sem 0.76 0.76 0.7774 time 9.10
Epoch 11 sem 0.75 0.76 0.7784 time 8.93
Epoch 12 sem 0.72 0.77 0.7788 time 9.40
Epoch 13 sem 0.69 0.78 0.7813 time 10.42
Epoch 14 sem 0.66 0.79 0.7778 time 9.60
Epoch 15 sem 0.66 0.79 0.7731 time 9.27
Epoch 16 sem 0.64 0.80 0.7885 time 8.89
Epoch 17 sem 0.61 0.81 0.7899 time 10.03
Epoch 18 sem 0.59 0.81 0.7771 time 9.93
Epoch 19 sem 0.58 0.82 0.7867 time 9.53
Epoch 20 sem 0.58 0.81 0.7810 time 9.41
Epoch 21 sem 0.55 0.82 0.7788 time 9.39
Epoch 22 sem 0.54 0.83 0.7764 time 8.97
Epoch 23 sem 0.51 0.84 0.7806 time 8.77
Epoch 24 sem 0.52 0.83 0.7831 time 9.50
Epoch 25 sem 0.50 0.84 0.7849 time 9.83
Epoch 26 sem 0.50 0.84 0.7881 time 10.09
Epoch 27 sem 0.50 0.84 0.7860 time 9.34
Epoch 28 sem 0.47 0.85 0.7874 time 9.05
Epoch 29 sem 0.47 0.85 0.7888 time 9.75
Epoch 30 sem 0.46 0.85 0.7903 time 10.05
Epoch 31 sem 0.44 0.86 0.7892 time 11.15
Epoch 32 sem 0.45 0.86 0.7885 time 9.90
Epoch 33 sem 0.42 0.87 0.7928 time 9.33
Epoch 34 sem 0.44 0.86 0.7863 time 10.34
Epoch 35 sem 0.41 0.87 0.7917 time 10.45
Epoch 36 sem 0.41 0.87 0.7910 time 11.78
Epoch 37 sem 0.39 0.87 0.7920 time 12.47
Epoch 38 sem 0.39 0.87 0.7914 time 11.76
Epoch 39 sem 0.38 0.88 0.7885 time 12.05
Epoch 40 sem 0.38 0.88 0.7991 time 11.84
Epoch 41 sem 0.37 0.88 0.7938 time 13.22
Epoch 42 sem 0.36 0.89 0.7828 time 11.33
Epoch 43 sem 0.37 0.89 0.7949 time 11.80
Epoch 44 sem 0.34 0.89 0.7970 time 12.02
Epoch 45 sem 0.36 0.89 0.7949 time 12.16
Epoch 46 sem 0.34 0.90 0.7928 time 11.49
Epoch 47 sem 0.35 0.89 0.7917 time 11.92
Epoch 48 sem 0.34 0.90 0.7917 time 11.36
Epoch 49 sem 0.34 0.89 0.7874 time 11.44
Done training, best_epoch: 40, best_acc: 0.7991
