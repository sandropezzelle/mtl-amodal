import argparse
import os
import pickle

import numpy as np
import pandas as pd
import unicodecsv as csv
from keras.models import load_model

if __name__ == "__main__":
    preprocessed_dataset_path = "lang_dataset/"
    embeddings_filename = "/mnt/povobackup/clic/sandro.pezzelle/corpus-and-vectors/GoogleNews-vectors-negative300.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dataset_path", type=str, default=preprocessed_dataset_path)
    parser.add_argument("--embeddings_filename", type=str, default=embeddings_filename)
    parser.add_argument("--model_filename", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    index_filename = os.path.join(args.preprocessed_dataset_path, "index.pkl")
    print("Loading filename: {}".format(index_filename))
    with open(index_filename, mode="rb") as in_file:
        index = pickle.load(in_file)
        token2id = index["token2id"]
        id2token = index["id2token"]
        m_out2id = index["m_out2id"]
        id2m_out = index["id2m_out"]
        r_out2id = index["r_out2id"]
        id2r_out = index["id2r_out"]

    test_filename = os.path.join(args.preprocessed_dataset_path, "test.pkl")
    print("Loading filename: {}".format(test_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "test.pkl"), mode="rb") as in_file:
        test = pickle.load(in_file)
        dataset_t = test["dataset_t"]
        t_m_out = test["t_m_out"]
        t_q_out = test["t_q_out"]
        t_r_out = test["t_r_out"]
        dataset_t_names = test["dataset_t_names"]
        dataset_t_years = test["dataset_t_years"]

    print("Loading filename: {}".format(args.model_filename))
    model = load_model(args.model_filename)

    print("Evaluating model...")
    scores = model.evaluate(dataset_t, t_m_out, batch_size=args.batch_size)
    for i in range(len(scores)):
        print("%s: %.4f%%" % (model.metrics_names[i], scores[i]))

    predictions = model.predict(dataset_t, batch_size=args.batch_size)

    y_pred_msl = np.argmax(predictions, axis=1)
    y_valarr_msl = np.argmax(t_m_out, axis=1)
    pd.crosstab(y_valarr_msl, y_pred_msl, margins=True).to_csv(args.model_filename.replace(".hdf5", ".confusion_msl"))

    with open(args.model_filename.replace(".hdf5", ".predictions"), mode="w") as out_file:
        writer = csv.writer(out_file, delimiter="\t", encoding="utf-8")
        writer.writerow(
            [
                "names",
                "years",
                "t_m_out",
                "t_pred_m_out",
            ]
        )
        for i, scenario in enumerate(dataset_t):
            print("Processing scenario {}/{}".format(i, len(dataset_t)))
            formatted_names = ", ".join([name for name in dataset_t_names[i] if name != "#pad#"])
            formatted_years = ", ".join([year for year in dataset_t_years[i] if year != "#pad#"])
            formatted_t_m_out = id2m_out[tuple(t_m_out[i])]
            formatted_t_pred_m_out = y_pred_msl[i]
            writer.writerow(
                [
                    formatted_names,
                    formatted_years,
                    formatted_t_m_out,
                    formatted_t_pred_m_out,
                ]
            )
