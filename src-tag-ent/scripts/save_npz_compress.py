import numpy as np

npy_file = "embed300.google.npy"
npz_file = "embed300.google.npz"

embed_tensor = np.load(npy_file)
print(embed_tensor.shape, embed_tensor.dtype)

np.savez_compressed(npz_file, word_embed=embed_tensor)
tensor_dict = np.load(npz_file)
print(tensor_dict['word_embed'].shape, tensor_dict['word_embed'].dtype)


