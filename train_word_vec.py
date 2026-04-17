from gensim.models import Word2Vec
from gensim import utils
from gensim.models import KeyedVectors
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize
import sys

def read_file(path):
    if 'token' in path
        with open(path, 'r', encoding='utf-8', newline='') as p:
            text = p.read()
            sentences = sent_tokenize(text)
            word_matrix = [word_tokenize(s.lower()) for s in sentences]
    else:
        with open(path, 'r', encoding='utf-8', newline='') as p:
            sent = []
            for line in p:
                line = line.replace('\n', '')
                sent.append(line)
            word_matrix = [word_tokenize(s.lower()) for s in sent]
    return word_matrix


if __name__ == "__main__":
    # outdir = "word2vec"
    path = sys.argv[1]
    output_path = sys.argv[2]
    # if not os.path.exists(outdir):
    #    os.mkdir(outdir)
        
    output_list = []
    
    #paths = glob.glob("/Users/marsk757/elliit/for_word_vec_training/*/*.txt")
    # path = glob.glob("semeval2020_ulscd_swe/corpus1/lemma/kubhist2a.txt")
    # path = 'kubhist2_token.txt'
    # print("nr of files:", len(path))
    
    sentences = read_file(path)
    output_list.extend(sentences)
        
        
    for line in output_list[:5]:
        print(line)

    
    print("Start training, with nr of sentences: ", len(output_list))
    model = Word2Vec(sentences=output_list, size=100, window=3, min_count=10, workers=4)
    word_vectors = model.wv
    word_vectors.save('word2vec_temporal_token.model')

    print('done')
    
    # model_loaded = KeyedVectors.load('word2vec.model')
    # print("\nantyda")
    # print("-------")
    # print(model_loaded.most_similar('antyda', topn=5))
    
    
    # print("\ngagn")
    # print("-------")
    # print(model_loaded.most_similar('gagn', topn=5))

   
