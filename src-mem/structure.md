# entities

* lexical, no pos: 0.64
* avg embed, pos : 0.4164
        + dropout: 0.4068
* avg embed, no pos: 0.3889

* max embed, pos : 0.4386

* conv entities, pos: 0.6125

# attention

* hop 1, word attention : 0.6186
  hop 3,                : 0.6900

* hop 1, input attention: 0.5961
  hop 3,                : 0.6046

* hop 1, multi attention: 0.7079
  hop 3,                : 0.7411

# label smoothing

  avg embed, pos
* smoothing: 0.4107
* no smoothing: 0.4154


# pos embedding

hop 1, multi attention

* pos embedding: 0.7079
* no pos embedding: 0.6839

# context or sentence

hop 1, multi attention

* context:  0.7079, 0.7136
* sentence: 0.6843
