from tensor2tensor.data_generators import tokenizer

doc = ["I've got it! Don't you see? :)", 
"TY KU /taɪkuː/ is an American alcoholic beverage company",
"The GOAT Store (Games Of All Type Store) LLC is one of",
"188BET is an online sportsbook provider. 188BET is owned by Cube Limited",
"Avista Utilities is a U.S. energy company.",
"it has 18ml.",
"''i've got it!!!"]

# for text in doc:
#   tokens = tokenizer.encode(text)
#   print(tokens)

import sys
import six
import unicodedata

_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))
n = len(_ALPHANUMERIC_CHAR_SET)
print(n)