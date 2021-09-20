import pickle
from pathlib import Path
import matplotlib.pyplot as plt


def plot(scores, mean_scores):
    plt.ion()
    plt.clf()
    plt.xlabel('n_games')
    plt.plot(scores, label="score")
    plt.plot(mean_scores, label="avg_score")
    plt.axhline(y=mean_scores[-1], color='orange', linestyle='--')
    plt.ylim(ymin=0)
    plt.legend(loc='upper left')


def save_plot(fpath):
    fpath.parent.mkdir(parents=True, exist_ok=True)
    i = 1 if Path.is_file(fpath) else -1
    try:
        name, ending = str(fpath).split(".")
    except ValueError:
        name = str(fpath)
        ending = ""

    while i >= 1:
        fpath = Path(name + "_" + str(i) + "." + ending)
        if Path.is_file(fpath):
            i += 1
        else:
            i = -1
    plt.savefig(fpath)
    print(f"Saved plot to '{fpath}'")


def dict_to_string(dict_, sep="\n"):
    """Converts dictionary to string.

    Each dictionary entry is converted as 'key: value' followed by seperator (sep).
    """
    as_string = ""
    for key in dict_:
        as_string += str(key) + ": " + str(dict_[key]) + sep
    return as_string


def write_to_file(data, fpath, text=False):
    Path.mkdir(fpath, parents=True, exist_ok=True)
    if text:
        with open(fpath, "wt") as file:
            file.write(data)
    else:
        with open(fpath, "wb") as file:
            pickle.dump(data, file)


def read_from_file(fpath, text=False):
    try:
        if text:
            with open(fpath, "rt") as file:
                lines = file.readlines()
                return lines
        else:
            with open(fpath, "rb") as file:
                data = pickle.load(file)
                return data
    except FileNotFoundError:
        return None
