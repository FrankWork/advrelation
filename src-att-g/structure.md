- input entity mh attention, sentence cnn + max pooling, concat
  acc: 79.05
- pool ent
  acc: 79.53, 79.91
- pool ent, pool max
  acc 80.49  f1 **84.28**
  acc 80.06, 80.99, 
  acc 79.70, 80.20
  acc 80.59 80.06

- pool ent, pool max, pool att
  acc 79.59
- pool ent, self attention inputs cnn max pool
  acc 79.91
- pool ent, pool max, pool max as input to pool att
  acc 80.28
- pool ent, pool max, pool ent as input to pool att
  acc 80.16
- pool ent, pool max, attentive pooling 
  acc 0.7944
- pool ent, pool att
  acc 80.05


# deep conv

- cnn + max pooling
  acc 79.38
- 2cnn + max pooling
  acc 79.24