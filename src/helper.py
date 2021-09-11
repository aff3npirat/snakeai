import pickle
from pathlib import Path
import matplotlib.pyplot as plt

plt.ion()


def plot(scores, mean_scores, plot_name=""):
    plt.clf()
    plt.title(plot_name)
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


def save_to_binary_file(data, fpath):
    fpath.parent.mkdir(parents=True, exist_ok=True)
    file = open(fpath, 'wb')
    pickle.dump(data, file)
    return True


def save_string_to_file(data, fpath):
    fpath.parent.mkdir(parents=True, exist_ok=True)
    file = open(fpath, 'w')
    file.write(data)
    return True


def read_from_binary_file(fpath):
    try:
        with open(fpath, "rb") as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        return None


def read_string_from_file(fpath):
    try:
        with open(fpath, "r") as file:
            data = file.readlines()
            return data
    except FileNotFoundError:
        return None


def array_to_byte(arr):
    """Converts an array to a single byte, where first element in array is MSB.
    A single bit equals 1 if corresponding element in array is interpreted as true.
    """
    byte = 0b0
    for b in arr:
        byte = byte << 1
        if b:
            byte |= 1
    return byte


def dict_to_string(dict, sep="\n"):
    """Converts dictionary to string.

    Each dictionary entry is converted as 'key: value' followed by seperator (sep).
    """
    as_string = ""
    for key in dict:
        as_string += str(key) + ": " + str(dict[key]) + sep
    return as_string
