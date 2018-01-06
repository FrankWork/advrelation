# dropout vs batch norm

* lexical only 0.64
* cnn with 310 filters: 0.7739
          + batch norm: 0.7307
          + input drop: 0.7871
   + input output drop: 0.7929
* cnn with 600 filters
   + input output drop:: 0.7957
* cnn with 100 filters
   + input output drop:: 0.7825

# position embedding

  input output dropout

* cnn 310 filters, 2*5 dim, concat: 0.7929
* cnn 300 filters, 300 dim, add   : 0.7796

# wide vs deep
 cnn 310 filters, input output dropout

* wide, kernel 3,4,5    : 0.7907
* deep, kernel 3, layer 3: 0.7975
* residual, kernel3, layer3, 1x1 short cut: 0.7979
* residual, kernel3, layer3, skip short cut: 0.7993
* deep 8 : 0.7755
* res 8: 0.7754


# residual_net

 with input output dropout, no batch norm

* num_blocks 1, block layer 1 : 0.7975
* num_blocks 1, block layer 1, no 1x1 proj : 0.7936, 0.7946
* num_blocks 1, block layer 1, 1x1 proj : 0.7896, 0.2921
