
def change_label(txt_filename, cln_filename, out_filename):
  txt_file = open(txt_filename)
  cln_file = open(cln_filename)
  out_file = open(out_filename, 'w')

  for txt_line in txt_file:
    cln_line = cln_file.readline()
    
    txt_tokens = txt_line.strip().lower().split()
    cln_tokens = cln_line.strip().split()

    out_tokens = []
    out_tokens.append(cln_tokens[0]) # label
    out_tokens.extend(txt_tokens[1:])  # entity position
    
    out_file.write(' '.join(out_tokens)+'\n')
  
  txt_file.close()
  cln_file.close()
  out_file.close()


train_txt = 'data/SemEval/train_nopos_ty=6.txt'
train_cln = 'data/SemEval/train.cln'
train_out = 'data/SemEval/train.cln.2'

change_label(train_txt, train_cln, train_out)

test_txt = 'data/SemEval/test_nopos_ty=6.txt'
test_cln = 'data/SemEval/test.cln'
test_out = 'data/SemEval/test.cln.2'

change_label(test_txt, test_cln, test_out)


