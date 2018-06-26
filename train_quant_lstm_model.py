import argparse
import os
import pickle

import atexit
import numpy as np
from keras.callbacks import EarlyStopping

import prop_lstm_model
import quant_lstm_model
from utils import MyModelCheckpoint, start_logger, stop_logger

if __name__ == "__main__":
    """
    it reads the parameters,
    initializes the hyperparameters,
    preprocesses the input,
    trains the model
    """
    preprocessed_dataset_path = "lang_dataset2/"
    embeddings_filename = "/mnt/povobackup/clic/sandro.pezzelle/corpus-and-vectors/GoogleNews-vectors-negative300.txt"
    weights_filename = "best_models/quant_lstm_model-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5"
    logging_filename = "best_models/train_quant_lstm_model.log"
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dataset_path", type=str, default=preprocessed_dataset_path)
    parser.add_argument("--embeddings_filename", type=str, default=embeddings_filename)
    parser.add_argument("--weights_filename", type=str, default=weights_filename)
    parser.add_argument("--logging_filename", type=str, default=logging_filename)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    start_logger(args.logging_filename)
    atexit.register(stop_logger)

    index_filename = os.path.join(args.preprocessed_dataset_path, "index.pkl")
    print("Loading filename: {}".format(index_filename))
    with open(index_filename, mode="rb") as in_file:
        index = pickle.load(in_file)
        token2id = index["token2id"]
        id2token = index["id2token"]
        # m_out2id = index["m_out2id"]
        # id2m_out = index["id2m_out"]
        # r_out2id = index["r_out2id"]
        # id2r_out = index["id2r_out"]

    train_filename = os.path.join(args.preprocessed_dataset_path, "train.pkl")
    print("Loading filename: {}".format(train_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "train.pkl"), mode="rb") as in_file:
        train = pickle.load(in_file)
        dataset_tr = train["dataset_tr"]
        tr_m_out = train["tr_m_out"]
        tr_q_out = train["tr_q_out"]
        tr_r_out = train["tr_r_out"]
        # dataset_tr_names = train["dataset_tr_names"]
        # dataset_tr_years = train["dataset_tr_years"]

    valid_filename = os.path.join(args.preprocessed_dataset_path, "valid.pkl")
    print("Loading filename: {}".format(valid_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "valid.pkl"), mode="rb") as in_file:
        valid = pickle.load(in_file)
        dataset_v = valid["dataset_v"]
        v_m_out = valid["v_m_out"]
        v_q_out = valid["v_q_out"]
        v_r_out = valid["v_r_out"]
        # dataset_v_names = valid["dataset_v_names"]
        # dataset_v_years = valid["dataset_v_years"]

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
    model = quant_lstm_model.QuantLSTMModel(embedding_matrix, token2id).build()
    checkpoint = MyModelCheckpoint(weights_filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    hist = model.fit(
        dataset_tr,
        tr_q_out,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        validation_data=(dataset_v, v_q_out),
        callbacks=[checkpoint]
    )
