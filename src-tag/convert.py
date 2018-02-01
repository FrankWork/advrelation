relations = []

tag_to_relation = {
'CE': 'Cause-Effect',
'CW': 'Component-Whole',
'CC': 'Content-Container',
'ED': 'Entity-Destination',
'EO': 'Entity-Origin',
'IA': 'Instrument-Agency',
'MC': 'Member-Collection',
'MT': 'Message-Topic',
'PP': 'Product-Producer',
'O': 'Other',
}
with open('tag_results.txt') as f:
  for line in f:
    seg = line.strip().split()
    if len(seg) == 0:
      relations.append('Other')
      continue

    tag = seg[0].split('-')
    if tag[1] == 'O':
      relations.append('Other')
      continue
    rel_str = tag_to_relation[tag[1]]
    rel_role = '(e1,e2)' if tag[-1]=='1' else '(e2,e1)'
    relations.append(rel_str+rel_role)

start_no = 8001

with open('results.txt', 'w') as f:
  for idx, rel in enumerate(relations):
    if idx < 2717:
      f.write('%d\t%s\n' % (start_no+idx, rel))


# <<< (2*9+1)-WAY EVALUATION (USING DIRECTIONALITY)>>>:

# Confusion matrix:
#         C-E1 C-E2 C-W1 C-W2 C-C1 C-C2 E-D1 E-D2 E-O1 E-O2 I-A1 I-A2 M-C1 M-C2 M-T1 M-T2 P-P1 P-P2  _O_ <-- classified as
#       +-----------------------------------------------------------------------------------------------+ -SUM- skip ACTUAL
#  C-E1 | 101   14    0    1    0    0    1    0    0    0    1    2    0    1    2    0    0    0   11 |  134    0  134
#  C-E2 |   7  162    0    0    0    0    0    0   11    0    0    0    0    2    0    0    0    3    9 |  194    0  194
#  C-W1 |   0    1  102   17    4    4    3    1    2    1    0    2    3    6    3    0    0    2   11 |  162    0  162
#  C-W2 |   0    1   24   85    0    1    2    0    0    1    0    5    0    2    2    2    0    2   23 |  150    0  150
#  C-C1 |   0    0    1    1  130    7    6    1    3    0    0    0    0    1    0    0    0    1    2 |  153    0  153
#  C-C2 |   0    0    0    4    1   26    1    0    2    1    0    0    0    1    0    0    0    0    3 |   39    0   39
#  E-D1 |   0    0    1    0   13    3  255    3    1    1    0    1    0    0    2    0    0    3    8 |  291    0  291
#  E-D2 |   0    0    0    0    0    0    0    1    0    0    0    0    0    0    0    0    0    0    0 |    1    0    1
#  E-O1 |   2    1    0    0    0    0    7    0  163    6    0    1    0    4    2    0    6    0   19 |  211    0  211
#  E-O2 |   1    0    1    0    1    1    0    0    7   27    0    1    0    0    0    0    0    1    7 |   47    0   47
#  I-A1 |   0    0    1    0    0    1    0    0    0    0    4    3    1    0    1    0    0    1   10 |   22    0   22
#  I-A2 |   1    1    3    8    0    0    1    0    2    1    4   76    0    1    3    2    0    6   25 |  134    0  134
#  M-C1 |   0    0    2    0    0    0    1    0    2    0    0    1   15    3    0    1    1    0    6 |   32    0   32
#  M-C2 |   0    0    6    3    1    0    0    0    1    0    1    0    9  161    2    1    0    5   11 |  201    0  201
#  M-T1 |   0    0    1    6    0    1    3    0    0    0    0    2    0    2  148   17    2    7   21 |  210    0  210
#  M-T2 |   0    0    0    0    0    0    0    0    0    0    0    1    0    0    8   31    2    3    6 |   51    0   51
#  P-P1 |   1    6    1    2    0    0    1    1    5    0    0    0    0    3    1    2   61   13   11 |  108    0  108
#  P-P2 |   0    0    1    4    0    0    3    0    2    2    0    8    0    2    2    3    8   71   17 |  123    0  123
#   _O_ |   3   12   15   34   22    3   28    2   36    7    2   18    4   37   31    7   10   20  163 |  454    0  454
#       +-----------------------------------------------------------------------------------------------+
#  -SUM-  116  198  159  165  172   47  312    9  237   47   12  121   32  226  207   66   90  138  363   2717    0 2717

# Coverage = 2717/2717 = 100.00%
# Accuracy (calculated for the above confusion matrix) = 1782/2717 = 65.59%
# Accuracy (considering all skipped examples as Wrong) = 1782/2717 = 65.59%
# Accuracy (considering all skipped examples as Other) = 1782/2717 = 65.59%

# Results for the individual relations:
#       Cause-Effect(e1,e2) :    P =  101/ 116 =  87.07%     R =  101/ 134 =  75.37%     F1 =  80.80%
#       Cause-Effect(e2,e1) :    P =  162/ 198 =  81.82%     R =  162/ 194 =  83.51%     F1 =  82.65%
#    Component-Whole(e1,e2) :    P =  102/ 159 =  64.15%     R =  102/ 162 =  62.96%     F1 =  63.55%
#    Component-Whole(e2,e1) :    P =   85/ 165 =  51.52%     R =   85/ 150 =  56.67%     F1 =  53.97%
#  Content-Container(e1,e2) :    P =  130/ 172 =  75.58%     R =  130/ 153 =  84.97%     F1 =  80.00%
#  Content-Container(e2,e1) :    P =   26/  47 =  55.32%     R =   26/  39 =  66.67%     F1 =  60.47%
# Entity-Destination(e1,e2) :    P =  255/ 312 =  81.73%     R =  255/ 291 =  87.63%     F1 =  84.58%
# Entity-Destination(e2,e1) :    P =    1/   9 =  11.11%     R =    1/   1 = 100.00%     F1 =  20.00%
#      Entity-Origin(e1,e2) :    P =  163/ 237 =  68.78%     R =  163/ 211 =  77.25%     F1 =  72.77%
#      Entity-Origin(e2,e1) :    P =   27/  47 =  57.45%     R =   27/  47 =  57.45%     F1 =  57.45%
#  Instrument-Agency(e1,e2) :    P =    4/  12 =  33.33%     R =    4/  22 =  18.18%     F1 =  23.53%
#  Instrument-Agency(e2,e1) :    P =   76/ 121 =  62.81%     R =   76/ 134 =  56.72%     F1 =  59.61%
#  Member-Collection(e1,e2) :    P =   15/  32 =  46.88%     R =   15/  32 =  46.88%     F1 =  46.88%
#  Member-Collection(e2,e1) :    P =  161/ 226 =  71.24%     R =  161/ 201 =  80.10%     F1 =  75.41%
#      Message-Topic(e1,e2) :    P =  148/ 207 =  71.50%     R =  148/ 210 =  70.48%     F1 =  70.98%
#      Message-Topic(e2,e1) :    P =   31/  66 =  46.97%     R =   31/  51 =  60.78%     F1 =  52.99%
#   Product-Producer(e1,e2) :    P =   61/  90 =  67.78%     R =   61/ 108 =  56.48%     F1 =  61.62%
#   Product-Producer(e2,e1) :    P =   71/ 138 =  51.45%     R =   71/ 123 =  57.72%     F1 =  54.41%
#                    _Other :    P =  163/ 363 =  44.90%     R =  163/ 454 =  35.90%     F1 =  39.90%

# Micro-averaged result (excluding Other):
# P = 1619/2354 =  68.78%     R = 1619/2263 =  71.54%     F1 =  70.13%

# MACRO-averaged result (excluding Other):
# P =  60.36%	R =  66.66%	F1 =  61.20%



# <<< (9+1)-WAY EVALUATION IGNORING DIRECTIONALITY >>>:

# Confusion matrix:
#          C-E  C-W  C-C  E-D  E-O  I-A  M-C  M-T  P-P  _O_ <-- classified as
#       +--------------------------------------------------+ -SUM- skip ACTUAL
#   C-E | 284    1    0    1   11    3    3    2    3   20 |  328    0  328
#   C-W |   2  228    9    6    4    7   11    7    4   34 |  312    0  312
#   C-C |   0    6  164    8    6    0    2    0    1    5 |  192    0  192
#   E-D |   0    1   16  259    2    1    0    2    3    8 |  292    0  292
#   E-O |   4    1    2    7  203    2    4    2    7   26 |  258    0  258
#   I-A |   2   12    1    1    3   87    2    6    7   35 |  156    0  156
#   M-C |   0   11    1    1    3    2  188    4    6   17 |  233    0  233
#   M-T |   0    7    1    3    0    3    2  204   14   27 |  261    0  261
#   P-P |   7    8    0    5    9    8    5    8  153   28 |  231    0  231
#   _O_ |  15   49   25   30   43   20   41   38   30  163 |  454    0  454
#       +--------------------------------------------------+
#  -SUM-  314  324  219  321  284  133  258  273  228  363   2717    0 2717

# Coverage = 2717/2717 = 100.00%
# Accuracy (calculated for the above confusion matrix) = 1933/2717 = 71.14%
# Accuracy (considering all skipped examples as Wrong) = 1933/2717 = 71.14%
# Accuracy (considering all skipped examples as Other) = 1933/2717 = 71.14%

# Results for the individual relations:
#              Cause-Effect :    P =  284/ 314 =  90.45%     R =  284/ 328 =  86.59%     F1 =  88.47%
#           Component-Whole :    P =  228/ 324 =  70.37%     R =  228/ 312 =  73.08%     F1 =  71.70%
#         Content-Container :    P =  164/ 219 =  74.89%     R =  164/ 192 =  85.42%     F1 =  79.81%
#        Entity-Destination :    P =  259/ 321 =  80.69%     R =  259/ 292 =  88.70%     F1 =  84.50%
#             Entity-Origin :    P =  203/ 284 =  71.48%     R =  203/ 258 =  78.68%     F1 =  74.91%
#         Instrument-Agency :    P =   87/ 133 =  65.41%     R =   87/ 156 =  55.77%     F1 =  60.21%
#         Member-Collection :    P =  188/ 258 =  72.87%     R =  188/ 233 =  80.69%     F1 =  76.58%
#             Message-Topic :    P =  204/ 273 =  74.73%     R =  204/ 261 =  78.16%     F1 =  76.40%
#          Product-Producer :    P =  153/ 228 =  67.11%     R =  153/ 231 =  66.23%     F1 =  66.67%
#                    _Other :    P =  163/ 363 =  44.90%     R =  163/ 454 =  35.90%     F1 =  39.90%

# Micro-averaged result (excluding Other):
# P = 1770/2354 =  75.19%     R = 1770/2263 =  78.21%     F1 =  76.67%

# MACRO-averaged result (excluding Other):
# P =  74.22%	R =  77.03%	F1 =  75.47%



# <<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:

# Confusion matrix:
#          C-E  C-W  C-C  E-D  E-O  I-A  M-C  M-T  P-P  _O_ <-- classified as
#       +--------------------------------------------------+ -SUM- xDIRx skip  ACTUAL
#   C-E | 263    1    0    1   11    3    3    2    3   20 |  307    21     0    328
#   C-W |   2  187    9    6    4    7   11    7    4   34 |  271    41     0    312
#   C-C |   0    6  156    8    6    0    2    0    1    5 |  184     8     0    192
#   E-D |   0    1   16  256    2    1    0    2    3    8 |  289     3     0    292
#   E-O |   4    1    2    7  190    2    4    2    7   26 |  245    13     0    258
#   I-A |   2   12    1    1    3   80    2    6    7   35 |  149     7     0    156
#   M-C |   0   11    1    1    3    2  176    4    6   17 |  221    12     0    233
#   M-T |   0    7    1    3    0    3    2  179   14   27 |  236    25     0    261
#   P-P |   7    8    0    5    9    8    5    8  132   28 |  210    21     0    231
#   _O_ |  15   49   25   30   43   20   41   38   30  163 |  454     0     0    454
#       +--------------------------------------------------+
#  -SUM-  293  283  211  318  271  126  246  248  207  363   2566   151     0   2717

# Coverage = 2717/2717 = 100.00%
# Accuracy (calculated for the above confusion matrix) = 1782/2717 = 65.59%
# Accuracy (considering all skipped examples as Wrong) = 1782/2717 = 65.59%
# Accuracy (considering all skipped examples as Other) = 1782/2717 = 65.59%

# Results for the individual relations:
#              Cause-Effect :    P =  263/( 293 +  21) =  83.76%     R =  263/ 328 =  80.18%     F1 =  81.93%
#           Component-Whole :    P =  187/( 283 +  41) =  57.72%     R =  187/ 312 =  59.94%     F1 =  58.81%
#         Content-Container :    P =  156/( 211 +   8) =  71.23%     R =  156/ 192 =  81.25%     F1 =  75.91%
#        Entity-Destination :    P =  256/( 318 +   3) =  79.75%     R =  256/ 292 =  87.67%     F1 =  83.52%
#             Entity-Origin :    P =  190/( 271 +  13) =  66.90%     R =  190/ 258 =  73.64%     F1 =  70.11%
#         Instrument-Agency :    P =   80/( 126 +   7) =  60.15%     R =   80/ 156 =  51.28%     F1 =  55.36%
#         Member-Collection :    P =  176/( 246 +  12) =  68.22%     R =  176/ 233 =  75.54%     F1 =  71.69%
#             Message-Topic :    P =  179/( 248 +  25) =  65.57%     R =  179/ 261 =  68.58%     F1 =  67.04%
#          Product-Producer :    P =  132/( 207 +  21) =  57.89%     R =  132/ 231 =  57.14%     F1 =  57.52%
#                    _Other :    P =  163/( 363 +   0) =  44.90%     R =  163/ 454 =  35.90%     F1 =  39.90%

# Micro-averaged result (excluding Other):
# P = 1619/2354 =  68.78%     R = 1619/2263 =  71.54%     F1 =  70.13%

# MACRO-averaged result (excluding Other):
# P =  67.91%	R =  70.58%	F1 =  69.10%



# <<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = 69.10% >>>