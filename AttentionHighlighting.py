import config
import word2vec.wor2vec as W2V

test_data = W2V.read_token_data(config.test_data_token_path)
test_data_n = W2V.read_data(config.test_data_path)
lex_adj = W2V.Read_lexicon(config.lexicon_adj)
lex_noun = W2V.Read_lexicon(config.lexicon_noun)
lex_verb = W2V.Read_lexicon(config.lexicon_verb)

print(len(lex_adj), len(lex_noun), len(lex_verb))

for idx, sent in enumerate(test_data[130:140]):
    lex_score = []
    for element in sent[0]:
        tags = element.split('/')
        if len(tags) == 2:
            voca = tags[0]
            pos = tags[1]
            if pos == 'Noun':
                lexicon = lex_noun
            elif pos == 'Adjective':
                lexicon = lex_adj
            elif pos == 'Verb':
                lexicon = lex_verb

            if lexicon.get(voca):
                score = abs(lexicon.get(voca))
            else:
                score = 0
        else:
            score = 0
        lex_score.append(score)

    doc_ = ''
    for i, element in enumerate(sent[0]):
        if lex_score[i] != 0:
            doc_ += ' ' + "\x1b[1;31m" + element + "\x1b[1;m"
        else:
            doc_ += ' ' + element

    lenn = 0
    doc = ""
    for token in test_data_n[idx+130][1].split(' '):
        posing = W2V.tokenize(token)
        flag = False
        for i in range(lenn, lenn + len(posing)):
            if lex_score[i] != 0:
                flag = True
                break

        if flag:
            doc += ' '+ "\x1b[1;31m" + token + "\x1b[1;m"
        else:
            doc += ' ' + token

        lenn += len(posing)

    print("[{}] ".format(test_data_n[130+idx][0])+doc)
    print(doc_)