import random
import numpy as np
from bilstm import *
import math
from basic_func import *


def rand_algo(model, data):

    to_be_labeled = []

    for i in range(10):
        ind = random.randint(0, len(data))
        sent, data = data[ind], np.concatenate((data[:ind], data[ind+1:]), axis=0)
        to_be_labeled.append([sent, ind])

    return to_be_labeled, data


def most_uncertain_instance(model, data):

    to_be_labeled = []
    uncertainty = []
    preds = model.predict(data)

    for i in range(len(preds)):
        tag_guesses = np.max(preds[i], axis=1)
        uncertainty.append(min(tag_guesses))

    inds = np.argpartition(uncertainty, 10)
    uncertainty.sort()
    # print(sum(uncertainty)/len(uncertainty), sum(uncertainty[:10])/10, sum(uncertainty[-10:])/len(uncertainty[-10:]))

    for i in range(10):
        to_be_labeled.append([data[inds[i]], inds[i]])

    to_be_removed = list(inds[:10])
    data = np.delete(data, to_be_removed, 0)

    return to_be_labeled, data


def most_uncertain_sentence(model, data):

    to_be_labeled = []
    uncertainty = []
    preds = model.predict(data)

    for i in range(len(preds)):
        tag_guesses = np.max(preds[i], axis=1)
        uncertainty.append(sum(tag_guesses))

    inds = np.argpartition(uncertainty, 10)
    # uncertainty.sort()
    # print(sum(uncertainty)/len(uncertainty), sum(uncertainty[:10])/10, sum(uncertainty[-10:])/len(uncertainty[-10:]))

    for i in range(10):
        to_be_labeled.append([data[inds[i]], inds[i]])

    to_be_removed = list(inds[:10])
    data = np.delete(data, to_be_removed, 0)


    return to_be_labeled, data


def most_uncertain_diff(model, data):
    to_be_labeled = []
    uncertainty = []
    preds = model.predict(data)

    for i in range(len(preds)):
        new_uncertainty = 0
        for j in range(len(preds[i])):
            temp_list = list(preds[i][j][:])
            max_value = max(temp_list)
            temp_list.remove(max_value)
            second_max_value = max(temp_list)
            new_uncertainty += max_value - second_max_value

        uncertainty.append(new_uncertainty)

    inds = np.argpartition(uncertainty, 10)

    for i in range(10):
        to_be_labeled.append([data[inds[i]], inds[i]])

    to_be_removed = list(inds[:10])
    data = np.delete(data, to_be_removed, 0)

    return to_be_labeled, data


def highest_entropy(model, data):
    to_be_labeled = []
    entropy = []
    preds = model.predict(data)

    for i in range(len(preds)):
        new_entropy = 0
        for j in range(len(preds[i])):
            for h in range(len(preds[i][j])):
                #
                new_entropy += preds[i][j][h]*math.log(preds[i][j][h], 2)

        entropy.append(new_entropy)

    inds = np.argpartition(entropy, 10)

    for i in range(10):
        to_be_labeled.append([data[inds[i]], inds[i]])

    to_be_removed = list(inds[:10])
    data = np.delete(data, to_be_removed, 0)

    return to_be_labeled, data


def split_data_committee(num_of_splits, data):

    new_dataset = []

    for i in range(num_of_splits):

        high_limit = len(data)*(i+1)//10
        low_limit = len(data)*i//10

        if i == 0:
            new_dataset.append(data[high_limit:])
        elif i == num_of_splits - 1:
            new_dataset.append(data[:low_limit])
        else:
            lowd = data[:low_limit]
            highd = data[high_limit:]
            s = np.concatenate((lowd, highd), axis=0)
            new_dataset.append(s)

    return new_dataset


def train_multiple_models(num_of_models, x_mult_data, y_mult_data, lang, x_test, y_test,
                          word_tokenizer, embedding_weights, labels):
    models = []
    accs = []
    for i in range(num_of_models):

        x_tot_data = x_mult_data[i]
        y_tot_data = y_mult_data[i]

        temp_model, new_acc = handle_bi_lstm(lang, x_tot_data, y_tot_data, x_test, y_test,
                                              word_tokenizer, embedding_weights, labels)
        models.append(temp_model)
        accs.append(new_acc)

    return models, accs


def calc_score(classifications):
    max_count = 0
    for i in range(len(classifications)):
        curr_count = classifications.count(classifications[i])
        if curr_count > max_count:
            max_count = curr_count
    return len(classifications) - max_count


def get_disagreement_score(models, data):

    preds = []
    to_be_labeled = []
    diss_score = []

    for i in range(len(models)):
        preds.append(models[i].predict(data))

    for i in range(len(preds[0])):
        temp = 0
        for j in range(len(preds[0][0])):
            model_pred = []
            for h in range(len(preds)):
                # print(preds[h][i][j])
                model_pred.append(np.argmax(preds[h][i][j]))
            temp += calc_score(model_pred)
        # if temp > 0:
        #     print(temp)
        diss_score.append(temp)

    inds = np.argpartition(diss_score, 10)

    for i in range(10):
        to_be_labeled.append([data[inds[i]], inds[i]])

    to_be_removed = list(inds[:10])
    data = np.delete(data, to_be_removed, 0)

    return to_be_labeled, data


def fake_oracle_agent(to_be_labeled, oracle_out):

    new_x = []  # sentences chosen to be labeled
    new_y = []  # oracle's labels
    inds = [] # indexes of sentences to be labeled

    for i in range(len(to_be_labeled)):
        new_x.append(to_be_labeled[i][0])
        new_y.append(oracle_out[to_be_labeled[i][1]])
        inds.append(to_be_labeled[i][1])

    oracle_out = np.delete(oracle_out, inds, 0)

    new_x, new_y = np.array(new_x), np.array(new_y)

    return new_x, new_y, oracle_out


def active_learning_routine(iterations, model, unlabeled_input, oracle_out, x_test, y_test, history, algo):

    for i in range(iterations):
        # choose instances to be labeled
        to_be_labeled, unlabeled_input = algo(model, unlabeled_input)
        # label those instances
        new_x, new_y, oracle_out = fake_oracle_agent(to_be_labeled, oracle_out)

        _, new_acc = update_model(new_x, new_y, x_test, y_test, model)
        history.append(new_acc)

    return history


def info_density_algo(data, labeled, word_embeddings, word_tokenizer):

    to_be_labeled = []
    unsim = []

    for i in range(len(data)):
        temp = 0
        for j in range(len(labeled)):
            temp += 1 - sent_sim(data[i], labeled[j], word_embeddings, word_tokenizer)

        unsim.append(temp)

    inds = np.argpartition(unsim, 10)

    for i in range(10):
        to_be_labeled.append([data[inds[i]], inds[i]])

    to_be_removed = list(inds[:10])
    data = np.delete(data, to_be_removed, 0)

    return to_be_labeled, data


def info_density_routine(iterations, model, unlabeled_input, oracle_out, x_test, y_test, history, labeled,
                         word_tokenizer, word_embeddings):

    for i in range(iterations):
        # choose instances to be labeled
        to_be_labeled, unlabeled_input = info_density_algo(unlabeled_input, labeled, word_embeddings, word_tokenizer)
        # label those instances
        new_x, new_y, oracle_out = fake_oracle_agent(to_be_labeled, oracle_out)

        labeled = np.concatenate((labeled, new_x), axis=0)
        _, new_acc = update_model(new_x, new_y, x_test, y_test, model)
        history.append(new_acc)

    return history


def query_by_committee(input_tr, out_tr, lang, input_test, out_test, word_tokenizer, embedding_weights, labels,
                       unlabeled_input, oracle_out):
    input_tr_mul = split_data_committee(10, input_tr)
    out_tr_mul = split_data_committee(10, out_tr)

    models, accs = train_multiple_models(10, input_tr_mul, out_tr_mul, lang, input_test,
                                         out_test, word_tokenizer, embedding_weights, labels)
    print(accs)
    history_of_accs = [accs]

    for i in range(20):
        to_be_labeled, unlabeled_input = get_disagreement_score(models, unlabeled_input)

        new_x, new_y, oracle_out = fake_oracle_agent(to_be_labeled, oracle_out)
        new_x_mul = split_data_committee(10, new_x)
        new_y_mul = split_data_committee(10, new_y)

        _, new_acc = update_multiple_models(new_x_mul, new_y_mul, input_test, out_test, models)
        history_of_accs.append(new_acc)

    return history_of_accs
