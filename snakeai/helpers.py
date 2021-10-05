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


def dict_to_str(dict_, sep="\n"):
    """Converts dictionary to string.

    Each dictionary entry is converted as 'key: value' followed by seperator (sep).
    """
    as_string = ""
    for key in dict_:
        as_string += str(key) + ": " + str(dict_[key]) + sep
    return as_string


def write_to_file(data, fpath, text=False):
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    if text:
        with open(fpath, "wt") as file:
            file.write(data)
    else:
        with open(fpath, "wb") as file:
            pickle.dump(data, file, protocol=5)


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


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="")
    # Print New Line on Complete
    if iteration == total:
        print()