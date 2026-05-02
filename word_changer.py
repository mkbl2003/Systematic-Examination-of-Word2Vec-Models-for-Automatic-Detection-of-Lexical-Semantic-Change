import re
def word_changer(targets, path, outputfile, suffix='_c1'):
    lines = []
    words = []
    with open(targets, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            line = line.replace('\n', '')
            words.append(line)
    with open(outputfile, 'w', encoding='utf-8', newline='') as w:
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b')
        with open(path, 'r', encoding='utf-8', newline='') as f:
            for line in f:
                line = line.replace('\n', '')
                updated_line = pattern.sub(r'\1' + suffix, line)
                w.write(updated_line)
                w.write('\n')
