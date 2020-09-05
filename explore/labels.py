#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data(path, ignore_col="sig_id", return_df=False):
    df = pd.read_csv(path)
    df.drop(columns=[ignore_col], inplace=True)

    if return_df:
        return df

    if df.shape[1] == 206:
        return df.to_numpy().astype(np.float32)

    return df.to_numpy()


def plotable(a):
    a = a.astype(float)
    a[a == 0] = np.nan
    return a


def main():
    y = read_data("data/train_targets_scored.csv", return_df=True)
    coo = y.T @ y

    # Check co-occurrence
    plt.matshow(plotable(coo.values))
    plt.xlabel("label number")
    plt.ylabel("label number")
    plt.savefig('coo.png')
    plt.show()

    # Check conditional probabilities
    interaction = coo.values.copy() / coo.values.sum(axis=1, keepdims=True)
    np.fill_diagonal(interaction, 0)
    nonzero = ~np.all(interaction == 0, axis=0)
    plt.matshow(plotable(interaction[np.ix_(nonzero, nonzero)]))
    plt.xlabel("interactiong label number")
    plt.ylabel("interactiong label number")
    plt.savefig('proba.png')
    plt.show()

    # Label frequency
    counts = y.sum(axis=0)
    significance = (-counts).argsort()
    x = np.arange(len(significance))
    plt.figure(figsize=(12, 6))
    plt.bar(x, counts[significance])
    plt.plot(x, x * 0 + 20, '--', color='black')
    plt.xticks(x, y.columns[significance], rotation=90)
    plt.ylabel("counts")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('counts.png')
    plt.show()


if __name__ == '__main__':
    main()
