import pandas as pd


def small_table(data):
    dfs = tables(data)
    max_df = []
    for decay_id in ["simple", "lin", "const"]:
        max_df.append(dfs[decay_id].max(0))
    return pd.DataFrame(max_df, index=["simple", "lin", "const"])


def tables(data):
    """
    Parameters
    ----------
    data : dict
        Dictionary containing evaluation data for single agent.

    Returns
    -------
    A dictionary containing a Dataframe for each eps-decay.
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
