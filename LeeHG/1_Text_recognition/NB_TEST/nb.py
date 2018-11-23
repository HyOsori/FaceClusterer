from konlpy.tag import Okt
from ckonlpy.tag import Postprocessor
from collections import Counter
from math import log


fw = open("traindata.txt", "w", encoding='utf-8')
out = open('output.txt', 'w', encoding='utf-8')
bad = open('bad.txt', 'w', encoding='utf-8')
good = open('good.txt', 'w', encoding='utf-8')

konl = Okt()
parcnt = 0
pos_dic = Counter()
neg_dic = Counter()
Valid_Count = 0
pos_cnum = 0
neg_cnum = 0
pos_wnum = 0
neg_wnum = 0


def parsing(line):
    global pos_dic
    global neg_dic
    global pos_cnum
    global neg_cnum
    global pos_wnum
    global neg_wnum
    global parcnt

    parcnt+=1
    if parcnt%1000==0:
        print(parcnt, 'comments done!')

    splited_line = line.split("\t")
    id = splited_line[0]
    comment = splited_line[1]
    label = int(splited_line[2])
    # print("parsing", id,"now...")
    morph = konl.pos(comment, norm=True)

    if label == 0:
        neg_cnum += 1
        for i in range(len(morph)):
            part = morph[i][1]
            if part == 'Noun' or part == 'Verb' or part == 'Adjective' \
                    or part == 'Exclamation' or part == 'KoreanParticle' \
                    or part == 'Alpha' or part == 'Foreign' or part == 'Number' or part == 'Punctuation':
                neg_wnum += 1
                neg_dic[morph[i][0]] += 1
                bad.write(morph[i][1]+'\t'+morph[i][0]+'\n')
    else:
        pos_cnum += 1
        for i in range(len(morph)):
            part = morph[i][1]
            if part == 'Noun' or part == 'Verb' or part == 'Adjective' \
                    or part == 'Exclamation' or part == 'KoreanParticle' \
                    or part == 'Alpha' or part == 'Foreign' or part == 'Number' or part == 'Punctuation':
                pos_wnum += 1
                pos_dic[morph[i][0]] += 1
                good.write(morph[i][1]+'\t'+morph[i][0]+'\n')


def classify(line, fnum):
    global pos_dic
    global neg_dic
    global pos_cnum
    global neg_cnum
    global pos_wnum
    global neg_wnum
    global Valid_Count

    splited_line = line.split("\t")
    id = splited_line[0]
    comment = splited_line[1]
    label = int(splited_line[2])
    morph = konl.pos(comment,norm=True)

    posval = 0
    negval = 0

    clen = len(morph)

    # P(words | positive) & P(words | negative)
    for i in range(clen):
        part = morph[i][1]
        word = morph[i][0]
        if part == 'Noun' or part == 'Verb' or part == 'Adjective' \
                or part == 'Exclamation' or part == 'KoreanParticle'\
                or part == 'Alpha' or part == 'Foreign' or part == 'Number' or part == 'Punctuation':
            pN = pos_dic[word]
            nN = neg_dic[word]
            posval += log((pN + 1) / (pos_wnum + fnum))
            negval += log((nN + 1) / (neg_wnum + fnum))

    # P(positive) & P(negative)
    posval += log(pos_cnum / (pos_cnum + neg_cnum))
    negval += log(neg_cnum / (pos_cnum + neg_cnum))


    if posval > negval:
        guess = 1
    else:
        guess = 0
    print('[' +str(guess)+ ']result: pos = ',str(posval),'neg = ',str(negval))
    if guess != int(label):
        # print(id)
        Valid_Count += 1

        out.write(id+'\t'+str(label)+'\t'+str(guess)+'\t'+str(posval)+'\t'+str(negval)+'\t')
        for i in range(len(morph)):
            out.write(morph[i][0]+" ")
        out.write('\n')


''' ==================
      Train the data
    =================='''
f = open("ratings_train.txt", "r", encoding='utf-8')
f.readline()    # read index
# cnt = 0
# while cnt<10000:
while True:
    # cnt+=1
    line = f.readline()
    if not line:
        break
    parsing(line)
f.close()
print("parsing done")

''' =========================
      Write before checking
    ========================='''
merge_dic = pos_dic.copy()
merge_dic.update(neg_dic)
total_word_num = len(merge_dic)

# Write parsed data
fw.write('word'+'\t'+'tag'+'\t'+'positive'+'\t'+'negative'+'\n')
for i in merge_dic.keys():
    fw.write(i+'\t\t')
    fw.write(str(pos_dic[i])+'\t'+str(neg_dic[i]))
    fw.write('\n')

# Write test info
out.write("--- 테스트 정보 ---\n")
out.write("전체 코멘트 개수 : "+str(pos_cnum+neg_cnum)+'\n')
out.write("긍정 코멘트 개수 : "+str(pos_cnum)+'\n')
out.write("부정 코멘트 개수 : "+str(neg_cnum)+'\n')
out.write("긍정 단어 수 : "+str(pos_wnum)+'\n')
out.write("부정 단어 수 : "+str(neg_wnum)+'\n')
out.write("중복 제외 단어 수 : "+str(total_word_num)+'\n')

''' =================
      checking start
    ================='''
print("start checking")

f = open("ratings_valid.txt", "r", encoding='utf-8')
f.readline()    # read index

while True:
    line = f.readline()
    if not line : break
    classify(line, total_word_num)

print('Wrong number is',Valid_Count)
f.close()
fw.close()
out.close()
'''
while True:
    test = input()
    if not test : break
    print(konl.pos(test))
    classify(line, total_word_num)
'''