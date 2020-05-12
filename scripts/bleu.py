from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import pdb

datafiles1 = [("./data_fr/valid1.txt", "v1"),("./data_fr/valid1_ref0.txt", "v1r0"), ("./data_fr/valid1_ref1.txt", "v1r1"), ("./data_fr/valid1_ref2.txt", "v1r2"), ("./data_fr/valid1_ref3.txt", "v1r3")]


datafiles2 = [("./data_fr/valid2.txt", "v2"),("./data_fr/valid2_ref0.txt", "v2r0"), ("./data_fr/valid2_ref1.txt", "v2r1"), ("./data_fr/valid2_ref2.txt", "v2r2"), ("./data_fr/valid2_ref3.txt", "v2r3")]

datafiles_test = [("./data_fr/test0.txt", "t0"), ("./data_fr/test0_ref0.txt", "t0r0"), ("./data_fr/test0_ref1.txt", "t0r1"), ("./data_fr/test0_ref2.txt", "t0r2"), ("./data_fr/test0_ref3.txt", "t0r3")]

output1 = [("./largelatent256_output/51_output_decoder_1_tran.txt", "d1tran"), ("./largelatent256_output/51_output_decoder_1_from.txt", "d1from"), ("./output_grammarly_basenochange/26_output_decoder_1_from.txt", "base1from"), ("./output_grammarly_basenochange/26_output_decoder_1_tran.txt", "base1tran")]

output2 = [("./largelatent256_output/51_output_decoder_2_tran.txt", "d2tran"), ("./largelatent256_output/51_output_decoder_2_from.txt", "d2from"), ("./output_grammarly_basenochange/26_output_decoder_2_from.txt", "base2from"), ("./output_grammarly_basenochange/26_output_decoder_2_tran.txt", "base2tran")]

# lines = [2250, 2262, 2263, 2277, 2312, 2319, 2255, 2256, 2281, 2285, 2299, 2227, 2240, 2248, 2288, 2317, 168, 539, 2254, 2297, 2308, 2313, 2221, 2243, 2245, 2246, 2270, 2293, 141, 816, 2315, 531, 699, 2233, 2251, 2258, 2260, 2273, 2276, 164, 195, 789, 2282, 2301, 2302, 2316, 2226, 2234, 2238, 2244]

lines1 = [922, 930, 931, 975, 935, 945, 946, 953, 981, 997]
lines1 = [lines1[i] - 1 for i in range(len(lines1))]

lines = [1231, 1243, 1244, 1258, 1293, 1300, 1236, 1237, 1262, 1266]
lines = [lines[i] - 1 for i in range(len(lines))]

lines2 = [2613, 2631, 2697, 2616, 2619, 2686, 2687, 2693, 2694, 2614] # lines from 26_output_decoder_1_from.txt for test0_ref0.txt
lines2 = [lines2[i] - 1 for i in range(len(lines2))]

lines3 = [2697, 2688, 2686, 2609, 2619, 2650, 2657, 2675, 2694, 2647]
lines3 = [lines3[i] - 1 for i in range(len(lines3))]

# read every line in 51_output_decoder_2_from.txt
# get corresponding matching references from valid2.txt

"""
:param: datafile is the actual file path like ./data_fr/valid2.txt
:param: output is the file path like ./largelatent256_output/51_output_decoder_2_from.txt
:param: lines is array of line numbers that output has from datafile
"""
def bleu_from(datafile, output, lines):
    file = open(datafile, 'r').readlines()
    valid2 = [file[i].lower().split() for i in lines]

    counter = 0
    references = []
    candidates = []

    with open(output, 'r') as fr: # _from.txt (informal)
        for line in fr:
            #pdb.set_trace()
            if counter >= len(valid2):
                break
            references.append([valid2[counter]])
            candidates.append(line.split())
            counter = counter + 1
    return corpus_bleu(references, candidates)

# read every line in 51_output_decoder_2_tran.txt 
# get corresponding references from valid2_ref0, valid2_ref1, valid2_ref2 ,valid2_ref3

"""
:param: datafiles should be a list of tuples of (filepath, names). this is not scalable lol
:param: output is a file path like ./largelatent256_output/51_output_decoder_2_tran.txt
:param: lines is array of line numbers that output has from datafile
"""
def bleu_tran(datafiles, output, lines):
    file1 = open(datafiles[1][0], 'r').readlines()
    file2 = open(datafiles[2][0], 'r').readlines()
    file3 = open(datafiles[3][0], 'r').readlines()
    file4 = open(datafiles[4][0], 'r').readlines()

    ref1 = [file1[i].lower().split() for i in lines]
    ref2 = [file2[i].lower().split() for i in lines]
    ref3 = [file3[i].lower().split() for i in lines]
    ref4 = [file4[i].lower().split() for i in lines]
    # formal reference translations

    counter = 0
    references = []
    candidates = []

    with open(output, 'r') as tran: # _tran.txt (formal)
        for line in tran:
            #pdb.set_trace()
            if counter >= len(ref1):
                break
            references.append([ref1[counter], ref2[counter], ref3[counter], ref4[counter]])
            candidates.append(line.split())
            counter = counter + 1

    return corpus_bleu(references, candidates)
"""
print("corpus bleu for 51_output_decoder_2_from: ")
print(bleu_from(datafiles2[0][0], output2[1][0], lines))

print("corpus bleu for 51_output_decoder_2_tran: ")
print(bleu_tran(datafiles2, output2[0][0], lines))

print("corpus bleu for 51_output_decoder_1_from: ")
print(bleu_from(datafiles1[0][0], output1[1][0], lines1))

print("corpus bleu for 51_output_decoder_1_tran: ")
print(bleu_tran(datafiles1, output1[0][0], lines1))
"""
# lines2 = [2613, 2631, 2697, 2616, 2619, 2686, 2687, 2693, 2694, 2614] # lines from 26_output_decoder_1_from.txt for test0_ref0.txt

print("corpus bleu for 26_output_decoder_1_from: ")
print(bleu_tran(datafiles_test, output1[2][0],lines2))

print("corpus bleu for 26_output_decoder_1_tran: ")
print(bleu_from(datafiles_test[0][0], output1[3][0], lines2))

print("corpus bleu for 26_output_decoder_2_from: ")
print(bleu_tran(datafiles_test, output2[2][0],lines3))

print("corpus bleu for 26_output_decoder_2_tran: ")
print(bleu_tran(datafiles_test, output2[3][0], lines3))