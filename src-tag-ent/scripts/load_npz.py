import numpy as np

npz_file = "embed300.google.npz"

tensor_dict = np.load(npz_file)
print(tensor_dict['word_embed'].shape, tensor_dict['word_embed'].dtype)

# 38s
