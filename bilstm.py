from keras.models import Sequential
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, Bidirectional


def bi_lstm_model(x_train, y_train, x_test, y_test, word_tokenizer,
                  embedding_weights, labels, lang):

    num_tags = len(labels)

    if lang == "eng":
        outd = 300
    else:
        outd = 96

    model = Sequential()
    # change output dim depending on language word2vec used, for english use 300, and for greek 96
    model.add(Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=outd,
                        input_length=100))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(TimeDistributed(Dense(num_tags+1, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    model_training = model.fit(x_train, y_train, batch_size=128, epochs=10)

    """
    preds = model.predict(x_test)
    corr = 0
    tots = 0
    for i in range(len(preds)):
        pred_tags = np.argmax(preds[i], axis=1)
        correct_tags = np.argmax(y_test[i], axis=1)
        for j in range(len(pred_tags)):
            tots += 1
            if pred_tags[j] == correct_tags[j]:
                corr += 1

    print(corr/tots, corr, tots)
    """

    _, accuracy = model.evaluate(x_test, y_test)
    print("Bidirectional LSTM model's accuracy:", accuracy)

    return model, model_training, accuracy


def handle_bi_lstm(lang, x_train, y_train, x_test, y_test, word_tokenizer, embedding_weights, labels):

    model, _, acc = bi_lstm_model(x_train, y_train,  x_test, y_test, word_tokenizer,
                                          embedding_weights, labels, lang)

    return model, acc


def update_model(x_train, y_train, x_test, y_test, model):

    new_model_training = model.fit(x_train, y_train, batch_size=128, epochs=10)

    _, accuracy = model.evaluate(x_test, y_test)
    print("Bidirectional LSTM model's accuracy:", accuracy)

    return new_model_training, accuracy


def update_multiple_models(x_train_mul, y_train_mul, x_test, y_test, models):

    models_training = []
    accs = []

    for i in range(len(models)):

        new_model_training = models[i].fit(x_train_mul[i], y_train_mul[i], batch_size=128, epochs=10)

        _, accuracy = models[i].evaluate(x_test, y_test)

        models_training.append(new_model_training)
        accs.append(accuracy)

    return models_training, accs

