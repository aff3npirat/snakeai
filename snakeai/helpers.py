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


def print_progress_bar(iteration,
                       total,
                       prefix='',
                       suffix='',
                       decimals=1,
                       length=100,
                       fill='#'
                       ):
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
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
        )
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'{prefix} |{bar}| {percent}% {suffix}', end="\r")
    if iteration == total:
        print()


def get_max_mean(data):
    """
    Parameters
    ----------
    data : dict
        Maps parameter settings to scores.
        E.g. '[0.1, 0.2]' -> [0, 1, 0, 10] would be the data from 4 runs.
    """
    n = len(list(data.values())[0])
    mean_scores = list(map(sum, data.values()))
    mean_scores = list(map(lambda x: x / n, mean_scores))
    max_mean = max(mean_scores)
    params = []
    for idx, val in enumerate(list(mean_scores)):
        if val == max_mean:
            params = list(data.keys())[idx][1:-1].split(", ")
            if len(params) == 2:
                eps, gamma = params
                params.append(f"[{eps:.3}, {gamma:.3}]")
            elif len(params) == 3:
                eps, gamma, m = params
                params.append(f"[{eps:.3}, {gamma:.3}, {m:.3}]")
    return max_mean, params


def small_table(data):
    """
    Parameters
    ----------
    data : dict
        Maps vision strings to nested dicts. Each vision string
        is mapped to a dictionary mapping decay strings to a dictionary which
        maps parameter settings to lists of scores.
        E.g. 'full' -> {
                        "simple": {'[0.0, 1.0]': [0, 10, 12, 5]},
                        "const": {'[0.0, 1.0]': [20, 25, 0, 30]}
                        }
    """
    col_labels = ["const", "simple", "lin"]
    row_labels = ["full", "partial", "diagonal", "short"]
    cell_text = []  # list of lists
    for vision in row_labels:
        row_text = []
        for decay in col_labels:
            mean, params = get_max_mean(data[vision][decay])
            row_text.append(f"max: {mean}\n{params}")
        cell_text.append(row_text)
    plt.ioff()
    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.scale(1, 2)
    return table


def eval_table(decay_id, data):
    # data[vision][decay][params]
    col_labels = ["full", "partial", "diagonal", "short"]
    row_labels = [param for param in data["full"][decay_id]]
    cell_text = [[f"{sum(scores) / len(scores):.2f}"
                  for vision in col_labels
                  if (scores := data[vision][decay_id][param]) is not None]
                 for param in row_labels]

    nrows = len(row_labels) + 1
    ncols = len(col_labels) + 1
    hcell, wcell = 0.1, 0.1
    hpad, wpad = 0.5, 0.5

    plt.ioff()
    fig, ax = plt.subplots(figsize=(ncols * wcell + wpad, nrows * hcell + hpad))
    ax.axis("off")
    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.scale(1, 2)
    plt.tight_layout()
    return table


def convert_data(data):
    """
    Parameters
    ----------
    data : dict
        Maps vision+decay strings to dictionaries mapping parameter settings to
        scores. E.g. "full+simple" -> {'[0.0, 1.0]': [0, 10, 20, 30]}
    """
    converted = {}
    for key in data:
        vision, decay = key.split("+")
        if vision in converted:
            converted[vision][decay] = data[key]
        else:
            converted[vision] = {decay: data[key]}
    return converted
