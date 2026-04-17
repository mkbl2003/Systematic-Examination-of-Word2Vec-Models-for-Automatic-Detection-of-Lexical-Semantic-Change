from adjustText import adjust_text
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re


def distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def read_targets():
    with open('semeval2020_ulscd_swe/targets.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def read_truths():
    truth_values = []
    with open('semeval2020_ulscd_swe/truth/binary.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = re.sub(r'\D', '', line.strip())
            truth_values.append(line)
    return truth_values


def get_splits(mode="original"):
    targets = read_targets()
    truths = read_truths()

    if mode == "original":
        train_idx = list(range(12)) + [14, 17, 18]
        test_idx = list(range(19, len(targets))) + [12, 13, 15, 16]

    elif mode == "last15":
        split_point = len(targets) - 15
        train_idx = list(range(split_point, len(targets)))
        test_idx = list(range(0, split_point))

    else:
        raise ValueError("Invalid mode. Use 'original' or 'last15'.")

    train_targets = [targets[i] for i in train_idx]
    train_truths = [truths[i] for i in train_idx]

    test_targets = [targets[i] for i in test_idx]
    test_truths = [truths[i] for i in test_idx]

    return train_targets, train_truths, test_targets, test_truths


def define_threshold(model1, model2, mode="original"):
    c1 = KeyedVectors.load(model1)
    c2 = KeyedVectors.load(model2)

    train_targets, train_truths, _, _ = get_splits(mode)

    pos = []
    neg = []

    for t, r in zip(train_targets, train_truths):
        try:
            d = distance(c1.get_vector(t), c2.get_vector(t))
            if r == '1':
                pos.append(d)
            else:
                neg.append(d)
        except KeyError:
            continue

    if not pos or not neg:
        raise ValueError("Not enough data to compute threshold.")

    return (np.mean(pos) + np.mean(neg)) / 2


def evaluation(model1, model2, threshold, mode="original"):
    c1 = KeyedVectors.load(model1)
    c2 = KeyedVectors.load(model2)

    _, _, test_targets, test_truths = get_splits(mode)

    guesses = []
    filtered_truths = []

    for t, r in zip(test_targets, test_truths):
        try:
            d = distance(c1.get_vector(t), c2.get_vector(t))
            pred = '1' if d >= threshold else '0'

            guesses.append(pred)
            filtered_truths.append(r)
        except KeyError:
            continue

    correct = sum(g == r for g, r in zip(guesses, filtered_truths))
    accuracy = correct / len(guesses)

    return f'Accuracy: {correct}/{len(guesses)} ({accuracy:.4f})'


def main(model1, model2, mode="original"):
    threshold = define_threshold(model1, model2, mode)
    return evaluation(model1, model2, threshold, mode)


def define_threshold_top_n_list(model1, model2, n=5, mode="original"):
    c1 = KeyedVectors.load(model1)
    c2 = KeyedVectors.load(model2)

    train_targets, train_truths, _, _ = get_splits(mode)

    pos = []
    neg = []

    for t, r in zip(train_targets, train_truths):
        try:
            words_c1 = {w for w, _ in c1.most_similar(t, topn=n)}
            words_c2 = {w for w, _ in c2.most_similar(t, topn=n)}

            overlap = len(words_c1 & words_c2)

            if r == '1':
                pos.append(overlap)
            else:
                neg.append(overlap)

        except KeyError:
            continue

    if not pos or not neg:
        raise ValueError("Not enough data to compute threshold.")

    return (np.mean(pos) + np.mean(neg)) / 2


def evaluation_top_n_list(model1, model2, threshold, n=5, mode="original"):
    c1 = KeyedVectors.load(model1)
    c2 = KeyedVectors.load(model2)

    _, _, test_targets, test_truths = get_splits(mode)

    guesses = []
    filtered_truths = []

    for t, r in zip(test_targets, test_truths):
        try:
            words_c1 = {w for w, _ in c1.most_similar(t, topn=n)}
            words_c2 = {w for w, _ in c2.most_similar(t, topn=n)}

            overlap = len(words_c1 & words_c2)

            pred = '1' if overlap <= threshold else '0'

            guesses.append(pred)
            filtered_truths.append(r)

        except KeyError:
            continue

    correct = sum(g == r for g, r in zip(guesses, filtered_truths))
    accuracy = correct / len(guesses)

    return f'Accuracy: {correct}/{len(guesses)} ({accuracy:.4f})'


def main_top_n_list(model1, model2, n=5, mode="original"):
    threshold = define_threshold_top_n_list(model1, model2, n, mode)
    return evaluation_top_n_list(model1, model2, threshold, n, mode)


def visualizer(model1, model2):
    c1 = KeyedVectors.load(model1)
    c2 = KeyedVectors.load(model2)

    targets = read_targets()
    filtered = [t for t in targets if t in c1.index_to_key and t in c2.index_to_key]

    vectors_c1 = c1[filtered]
    vectors_c2 = c2[filtered]

    all_vectors = np.vstack([vectors_c1, vectors_c2])

    pca = PCA(n_components=2)
    result = pca.fit_transform(all_vectors)

    labels = [f"{w}_c1" for w in filtered] + [f"{w}_c2" for w in filtered]
    colors = ['green'] * len(filtered) + ['blue'] * len(filtered)

    plt.figure(figsize=(10, 8))
    plt.scatter(result[:, 0], result[:, 1], c=colors)

    texts = []
    for i, label in enumerate(labels):
        texts.append(plt.text(result[i, 0], result[i, 1], label, fontsize=9))

    adjust_text(texts)

    plt.title("Word Embeddings Visualization - Token Models")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()
