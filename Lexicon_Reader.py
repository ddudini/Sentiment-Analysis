class Lex_Reader():
    def __init__(self):
        None

    def read_data(self, filename):
        dict = {}
        f = open(filename, 'r', encoding='utf-8')

        while True:
            line = f.readline()
            if not line: break
            tokens = line.split('\t')
            voca = tokens[0].split('/')[0]
            dict[voca] = float(tokens[2])
        f.close()

        return dict