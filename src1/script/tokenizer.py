import re

# nltk.tokenize.regexp.WordPunctTokenizer
# regexp = re.compile(r'\w+|[^\w\s]+')
regexp = re.compile(r'\d+|\w+|[^\w\s]')

def wordpunct_tokenizer(line):
  line = line.lower()
  line = re.sub(r"[^A-Za-z0-9(),.!?\'\`\"]", " ", line)
  line = re.sub(r'!+', '!', line)
  line = re.sub(r'\?+', '?', line)
  line = re.sub(r'[!\?]{2,}', '?', line)
  line = re.sub(r"\s{2,}", " ", line)
  return regexp.findall(line)

if __name__ == '__main__':
  line = "''and i've done it! In 1900's i have 20ml(ml), I said\"hello\"!!!????!!!"
  tokens = wordpunct_tokenizer(line)
  print(' '.join(tokens))

  line = "you're good(well)."
  tokens = wordpunct_tokenizer(line)
  print(' '.join(tokens))