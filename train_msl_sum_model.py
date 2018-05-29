import argparse
import os
import pickle

import numpy as np
from sklearn.metrics.classification import confusion_matrix

import msl_lstm_model
import msl_sum_model
from utils import MyModelCheckpoint

if __name__ == "__main__":
    """
    it reads the parameters,
    initializes the hyperparameters,
    preprocesses the input,
    trains the model
    """
    preprocessed_dataset_path = "lang_dataset/"
    embeddings_filename = "/mnt/povobackup/clic/sandro.pezzelle/corpus-and-vectors/GoogleNews-vectors-negative300.txt"
    weights_filename = "best_models/msl_lstm_model-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dataset_path", type=str, default=preprocessed_dataset_path)
    parser.add_argument("--embeddings_filename", type=str, default=embeddings_filename)
    parser.add_argument("--weights_filename", type=str, default=weights_filename)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    index_filename = os.path.join(args.preprocessed_dataset_path, "index.pkl")
    print("Loading filename: {}".format(index_filename))
    with open(index_filename, mode="rb") as in_file:
        index = pickle.load(in_file)
        token2id = index["token2id"]
        id2token = index["id2token"]

    train_filename = os.path.join(args.preprocessed_dataset_path, "train.pkl")
    print("Loading filename: {}".format(train_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "train.pkl"), mode="rb") as in_file:
        train = pickle.load(in_file)
        dataset_tr = train["dataset_tr"]
        tr_m_out = train["tr_m_out"]
        tr_q_out = train["tr_q_out"]
        tr_r_out = train["tr_r_out"]

    test_filename = os.path.join(args.preprocessed_dataset_path, "test.pkl")
    print("Loading filename: {}".format(test_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "test.pkl"), mode="rb") as in_file:
        test = pickle.load(in_file)
        dataset_t = test["dataset_t"]
        t_m_out = test["t_m_out"]
        t_q_out = test["t_q_out"]
        t_r_out = test["t_r_out"]

    valid_filename = os.path.join(args.preprocessed_dataset_path, "valid.pkl")
    print("Loading filename: {}".format(valid_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "valid.pkl"), mode="rb") as in_file:
        valid = pickle.load(in_file)
        dataset_v = valid["dataset_v"]
        v_m_out = valid["v_m_out"]
        v_q_out = valid["v_q_out"]
        v_r_out = valid["v_r_out"]

    print("Loading filename: {}".format(args.embeddings_filename))
    embeddings_index = {}
    with open(args.embeddings_filename) as in_file:
        for line in in_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(token2id) + 1, 300))
    for word, i in token2id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("Training model...")
    model = msl_sum_model.MSLSumModel(embedding_matrix, token2id).build()
    checkpoint = MyModelCheckpoint(weights_filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    hist = model.fit(
        dataset_tr,
        tr_m_out,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        validation_data=(dataset_v, v_m_out),
        callbacks=[checkpoint]
    )

    print("Evaluating model...")
    best_model = msl_sum_model.MSLSumModel(embedding_matrix, token2id).build()
    best_model.load_weights(checkpoint.last_saved_filename)
    scores = best_model.evaluate(dataset_t, t_m_out, batch_size=args.batch_size)
    print("%s: %.4f%%" % (model.metrics_names[1], scores[1]))

    y_pred = model.predict_classes(dataset_t, verbose=1)
    y_predarr = np.asarray(y_pred)
    y_valarr = np.array(t_m_out, dtype=np.float16)
    print("Confusion matrix:")
    print(confusion_matrix(y_valarr, y_predarr))
