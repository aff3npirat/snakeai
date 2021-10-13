import pandas as pd

from . import helpers


def small_table(data, peak=False):
    """
    Parameters
    ----------
    data : dict
        Containing evaluation data for single agent.
    peak : bool, optional
        If true table cells contain maximum score, if false cells contain mean
        score.
    """
    if peak:
        dfs = max_tables(data)
    else:
        dfs = mean_tables(data)
    max_df = []
    for decay_id in ["simple", "lin", "const"]:
        max_df.append(dfs[decay_id].max(0))
    return pd.DataFrame(max_df, index=["simple", "lin", "const"])


def mean_tables(data):
    """
    Parameters
    ----------
    data : dict
        Containing evaluation data for single agent.

    Returns
    -------
    A dictionary containing a DataFrame for each eps-decay.
    """
    # data[vision][decay][params]
    dfs = {}
    col_labels = ["full", "partial", "diagonal", "short"]
    for decay_id in ["simple", "const", "lin"]:
        row_labels = []
        for param_id in data["full"][decay_id]:
            params = param_id[1:-1].split(", ")
            if len(params) == 2:
                eps, gamma = params
                row_labels.append(f"[{eps:.3}, {gamma:.3}]")
            else:
                eps, gamma, m = params
                row_labels.append(f"[{eps:.3}, {gamma:.3}, {m:.3}]")

        rows = [[f"{sum(scores) / len(scores):.2f}"
                 for vision in col_labels
                 if (scores := data[vision][decay_id][param]) is not None]
                for param in data["full"][decay_id]]
        dfs[decay_id] = pd.DataFrame(rows,
                                     index=row_labels,
                                     columns=col_labels,
                                     dtype=float)
    return dfs


def max_tables(data):
    dfs = {}
    col_labels = ["full", "partial", "diagonal", "short"]
    for decay_id in ["simple", "const", "lin"]:
        row_labels = []
        for param_id in data["full"][decay_id]:
            params = param_id[1:-1].split(", ")
            if len(params) == 2:
                eps, gamma = params
                row_labels.append(f"[{eps:.3}, {gamma:.3}]")
            else:
                eps, gamma, m = params
                row_labels.append(f"[{eps:.3}, {gamma:.3}, {m:.3}]")

        rows = [[f"{max(scores):.2f}"
                 for vision in col_labels
                 if (scores := data[vision][decay_id][param]) is not None]
                for param in data["full"][decay_id]]
        dfs[decay_id] = pd.DataFrame(rows,
                                     index=row_labels,
                                     columns=col_labels,
                                     dtype=float)
    return dfs


def convert_data(data):
    """
    Parameters
    ----------
    data : dict
        Containing evaluation data for single agent.
    """
    converted = {}
    for key in data:
        vision, decay = key.split("+")
        if vision in converted:
            converted[vision][decay] = data[key]
        else:
            converted[vision] = {decay: data[key]}
    return converted


def html_tables(data, root_dir):
    """Save data as html tables.
    
    Creates four files (simple.html, const.html, lin.html, max.html) in
    specified root directory. Each file contains a table representing the peak
    and mean performances.
    
    Parameters
    ----------
    data : dict
        Evaluation data for single agent.
    root_dir : str
        Relative path to directory where tables should be saved to.
    """
    mean_dfs = mean_tables(data)
    max_dfs = max_tables(data)
    mean_dfs["max"] = small_table(data, peak=False)
    max_dfs["max"] = small_table(data, peak=True)
    for key in mean_dfs:
        df = pd.concat([mean_dfs[key], max_dfs[key]],
                       axis=1,
                       keys=["mean", "max"])
        html_table = df.to_html()
        helpers.write_to_file(html_table, f"{root_dir}/{key}.html", text=True)
