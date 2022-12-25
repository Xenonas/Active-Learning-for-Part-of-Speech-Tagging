import numpy as np
from keras.utils import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
import spacy


def create_reference_list(data_set):
    ref = []
    no_words = 0
    for i in range(0, len(data_set)):
        for j in range(0, len(data_set[i])):
            no_words += 1
            if not data_set[i][j][1] in ref:
                ref.append(data_set[i][j][1])
    return ref


def read_data(filename):
    data_file = open(filename, "r", encoding="utf-8")
    data = data_file.read()
    lines = data.split("\n\n")
    for i in range(0, len(lines)):
        sentence_anal = lines[i].split("\n")
        lines[i] = []
        for j in range(0, len(sentence_anal)):
            if sentence_anal[j]:
                if sentence_anal[j][0] != '#':
                    sent = sentence_anal[j].split("\t")
                    lines[i].append((sent[1], sent[3]))
    return lines


def split_data(dataset):
    # we are gonna be splitting the dataset into the input sentences and the output corresponding tags
    x_data = []
    y_data = []
    for sentence in dataset:
        temp_x = []
        temp_y = []
        for word in sentence:
            temp_x.append(word[0])
            temp_y.append(word[1])
        x_data.append(temp_x)
        y_data.append(temp_y)

    return x_data, y_data


def word2vec_emb(word_tokenizer):
    path = r'GoogleNews-vectors-negative300.bin'
    
    word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
    embedding_weights = np.zeros((len(word_tokenizer.word_index) + 1, 300))
    word2id = word_tokenizer.word_index

    for word, index in word2id.items():
        if index != 1 and word in word2vec:
            embedding_weights[index, :] = word2vec[word]

    return embedding_weights


def word2vec_gr(word_tokenizer):

    nlp = spacy.load('el_core_news_sm')
    embedding_weights = np.zeros((len(word_tokenizer.word_index) + 1, 96))
    word2id = word_tokenizer.word_index

    for word, index in word2id.items():
        if index != 1:
            embedding_weights[index, :] = nlp(word).vector

    return embedding_weights


def create_val(x_train, y_train, cut_off):
    cut_off = int(cut_off*len(x_train))

    x_val = x_train[:cut_off]
    y_val = y_train[:cut_off]

    x_train = x_train[cut_off:]
    y_train = y_train[cut_off:]

    return x_train, y_train, x_val, y_val


def preprocess_data(data, word_tokenizer=None, tag_tokenizer=None, embedding_weights=None, language="eng"):
    x_data, y_data = split_data(data)

    if word_tokenizer is None:
        word_tokenizer = Tokenizer(oov_token=True)
        word_tokenizer.fit_on_texts(x_data)
    x = word_tokenizer.texts_to_sequences(x_data)

    if tag_tokenizer is None:
        tag_tokenizer = Tokenizer()
        tag_tokenizer.fit_on_texts(y_data)
    y = tag_tokenizer.texts_to_sequences(y_data)

    x = pad_sequences(x, maxlen=100, padding="pre", truncating="post")
    y = pad_sequences(y, maxlen=100, padding="pre", truncating="post")

    if embedding_weights is None:
        if language == "eng":
            embedding_weights = word2vec_emb(word_tokenizer)
        else:
            embedding_weights = word2vec_gr(word_tokenizer)

    y = to_categorical(y)

    return x, y, word_tokenizer, tag_tokenizer, embedding_weights


def preprocess_input_sent(sentence, word_tokenizer):

    x = word_tokenizer.texts_to_sequences(sentence)

    x = pad_sequences(x, maxlen=100, padding="pre", truncating="post")

    return x


def show_res(training, name='Accuracy (across labeled batches)'):
    plt.plot(training)
    plt.title(name)
    plt.ylabel('accuracy')
    plt.xlabel('batches of sentences (1 batch = 10 sentences)')
    plt.show()


def show_res_crf(training, name='Accuracy (across epochs)'):
    plt.plot(training.history['viterbi_accuracy'])
    plt.plot(training.history['val_viterbi_accuracy'])
    plt.title(name)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'])
    plt.show()


def make_prediction(model, word, tag, sentence, labels):
    st = tag.texts_to_sequences([labels])
    decode_dict = {}
    for i in range(len(labels)):
        decode_dict[st[0][i]] = labels[i]
    sent = [sentence.split(" ")]
    pr = preprocess_input_sent(sent, word)
    mpla = model.predict(pr)
    l = len(sent[0])
    for i in range(len(mpla)):
        for j in range(len(mpla[i])):
            mpla[i][j][0] = -0.1
    p = np.argmax(mpla[0], axis=1)[-l:]
    res = []
    for i in range(len(p)):
        print(sent[0][i], ":", decode_dict[p[i]])
        res.append(decode_dict[p[i]])
    return res


def get_dataset(lang):

    if lang == "eng":
        path = r"venv/UD_English-EWT/en_ewt-ud-"
    elif lang == "gr":
        path = r"venv/UD_Greek-GDT/el_gdt-ud-"

    train = read_data(path + "train.conllu")
    test = read_data(path + "test.conllu")

    return train, test


def find_seq_cutoff(sent):
    for i in range(len(sent)):
        if sent[i] != 0:
            return i


def find_sent_vec(sent1, word_embeddings, word_tokenizer):

    length1 = find_seq_cutoff(sent1)
    try:
        real_sent1 = word_tokenizer.sequences_to_texts([sent1[length1:]])[0]
    except TypeError:
        return None

    if 'apple' in word_embeddings:
        set1 = word_embeddings["apple"] - word_embeddings["apple"]
    else:
        set1 = word_embeddings["του"] - word_embeddings["του"]
    for word_index in range(0, len(real_sent1)):
        word = real_sent1[word_index]
        if word in word_embeddings:
            set1 += word_embeddings[real_sent1[word_index]]
        else:
            length1 -= 1
    if length1 == 0:
        return set1
    set1 = set1/length1

    return set1


def sent_sim(sent1, sent2, word_embeddings, word_tokenizer):

    vec1 = find_sent_vec(sent1, word_embeddings, word_tokenizer)
    vec2 = find_sent_vec(sent2, word_embeddings, word_tokenizer)

    if vec1 is None or vec2 is None:
        return 1

    try:
        cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except RuntimeWarning:
        return 1

    return cosine_similarity

