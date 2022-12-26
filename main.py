from al_func import *
from basic_func import *


if __name__ == '__main__':

    lang_choice = input("Choose language (1) English or (2) Greek:")
    if lang_choice == "1":
        lang = "eng"
    else:
        lang = "gr"

    train, test = get_dataset(lang)
    labels = create_reference_list(train)

    print("Started preprocessing...")
    input_tr, out_tr, word_tokenizer, tag_tokenizer, embedding_weights = preprocess_data(train, language=lang)
    input_test, out_test, word_tokenizer, tag_tokenizer, embedding_weights \
        = preprocess_data(test, word_tokenizer, tag_tokenizer, embedding_weights, language=lang)
    print("Finished preprocessing!")

    input_tr, unlabeled_input = input_tr[500:600], np.concatenate((input_tr[:190], input_tr[300:]), axis=0)
    out_tr, oracle_out = out_tr[500:600], np.concatenate((out_tr[:190], out_tr[300:]), axis=0)

    algo_choice = input("Choose active learning method (1) Uncertainty Sampling or (2) Query by committee or "
                        "(3) Information Density")
    if algo_choice == '1':
        unc_choice = input("Choose uncertainty sampling algorithm (1) Random (2) Most uncertain instance "
                           "(3) Most uncertain sentence (4) Most uncertain sentence by probability difference or "
                           "(5) Highest entropy")
        final_model, new_acc = handle_bi_lstm(lang, input_tr, out_tr, input_test, out_test,
                                              word_tokenizer, embedding_weights, labels)
        accuracy_history = [new_acc]
        if unc_choice == '1':
            accuracy_history = active_learning_routine(20, final_model, unlabeled_input, oracle_out, input_test,
                                                       out_test, accuracy_history, rand_algo)
        elif unc_choice == '2':
            accuracy_history = active_learning_routine(20, final_model, unlabeled_input, oracle_out, input_test,
                                                       out_test, accuracy_history, most_uncertain_instance)
        elif unc_choice == '3':
            accuracy_history = active_learning_routine(20, final_model, unlabeled_input, oracle_out, input_test,
                                                       out_test, accuracy_history, most_uncertain_sentence)
        elif unc_choice == '4':
            accuracy_history = active_learning_routine(20, final_model, unlabeled_input, oracle_out, input_test,
                                                       out_test, accuracy_history, most_uncertain_diff)
        else:
            accuracy_history = active_learning_routine(20, final_model, unlabeled_input, oracle_out, input_test,
                                                       out_test, accuracy_history, highest_entropy)

        print(accuracy_history)
        show_res(accuracy_history)

    elif algo_choice == '2':
        history = query_by_committee(input_tr, out_tr, lang, input_test, out_test, word_tokenizer,
                                     embedding_weights, labels, unlabeled_input, oracle_out)
        print(history)
    else:
        if lang == "eng":
            path = 'GoogleNews-vectors-negative300.bin'
            word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
        else:
            nlp = spacy.load('el_core_news_sm')
            word2vec = {}
            for i in range(len(train)):
                for j in range(len(train[i])):
                    word2vec[train[i][j][0]] = nlp(train[i][j][0]).vector

        final_model, new_acc = handle_bi_lstm(lang, input_tr, out_tr, input_test, out_test,
                                              word_tokenizer, embedding_weights, labels)
        accuracy_history = [new_acc]
        accuracy_history = info_density_routine(20, final_model, unlabeled_input, oracle_out, input_test,
                                                out_test, accuracy_history, input_tr, word_tokenizer, word2vec)
        print(accuracy_history)
        show_res(accuracy_history)

    # INFORMATION DENSITY
    # accuracy_history = info_density_routine(20, final_model, unlabeled_input, oracle_out, input_test,
    #                                         out_test, accuracy_history, input_tr, word_tokenizer, word2vec)

    # accuracy_history = active_learning_routine(20, final_model, unlabeled_input, oracle_out, input_test,
    #                                            out_test, accuracy_history, highest_entropy)
