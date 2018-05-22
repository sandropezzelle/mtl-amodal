import argparse
import os
import pickle
import warnings

import numpy as np
from keras.callbacks import ModelCheckpoint

import msl_lstm_model

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
    predictions_filename = "best_models/msl_lstm_model-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.predictions"
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dataset_path", type=str, default=preprocessed_dataset_path)
    parser.add_argument("--embeddings_filename", type=str, default=embeddings_filename)
    parser.add_argument("--weights_filename", type=str, default=weights_filename)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open(os.path.join(args.preprocessed_dataset_path, "index.pkl"), mode="rb") as in_file:
        index = pickle.load(in_file)
        token2id = index["token2id"]
        id2token = index["id2token"]

    with open(os.path.join(args.preprocessed_dataset_path, "train.pkl"), mode="rb") as in_file:
        train = pickle.load(in_file)
        dataset_tr = train["dataset_tr"]
        tr_m_out = train["tr_m_out"]
        tr_q_out = train["tr_q_out"]
        tr_r_out = train["tr_r_out"]

    with open(os.path.join(args.preprocessed_dataset_path, "test.pkl"), mode="rb") as in_file:
        test = pickle.load(in_file)
        dataset_t = test["dataset_t"]
        t_m_out = test["t_m_out"]
        t_q_out = test["t_q_out"]
        t_r_out = test["t_r_out"]

    with open(os.path.join(args.preprocessed_dataset_path, "valid.pkl"), mode="rb") as in_file:
        valid = pickle.load(in_file)
        dataset_v = valid["dataset_v"]
        v_m_out = valid["v_m_out"]
        v_q_out = valid["v_q_out"]
        v_r_out = valid["v_r_out"]

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

    model = msl_lstm_model.MSLLSTMModel(embedding_matrix, token2id).build()


    class MyModelCheckpoint(ModelCheckpoint):
        def __init__(self, *args, **kwargs):
            super(MyModelCheckpoint, self).__init__(*args, **kwargs)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
                self.filepath = filepath
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch + 1, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve' %
                                      (epoch + 1, self.monitor))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)


    checkpoint = MyModelCheckpoint(weights_filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    hist = model.fit(
        dataset_tr,
        tr_m_out,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        validation_data=(dataset_v, v_m_out),
        callbacks=[checkpoint]
    )
    print(checkpoint.filepath)

    best_model = msl_lstm_model.MSLLSTMModel(embedding_matrix, token2id).build()
    best_model.load_weights(weights_filename)
    scores = best_model.evaluate(dataset_t, t_m_out, batch_size=args.batch_size)
    print("%s: %.4f%%" % (model.metrics_names[1], scores[1] * 100))
