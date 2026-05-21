import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
from matplotlib import colormaps
from scipy.stats import kendalltau, pearsonr, spearmanr, wilcoxon, mannwhitneyu, beta

from dynamical import DYNAMICAL_PROPERTIES
from network import NETWORK_PROPERTIES
from phi import PHASES

COLORBREWER = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams['figure.dpi'] = 600  # on-screen resolution
mpl.rcParams['savefig.dpi'] = 600  # saved output resolution


def _flush_plot(name):
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", name))
    plt.close()


def _load_all_data(directory="output", reduce=True, exp="ass", algorithm="single"):
    data = None
    for file in os.listdir(directory):
        if not file.endswith("csv"):  # or algorithm not in file:
            continue
        d = pd.read_csv(os.path.join(directory, file), sep=",", skiprows=1)
        task = file.split(".")[1]
        d["seed"] = int(file.split(".")[0])
        d["idx"] = "-".join(task.split("-")[:-1]) if len(task.split("-")) > 3 else task
        d["any_broken"] = any(d["is_broken"])
        d["algorithm"] = file.split(".")[2].split("-")[0]
        d["i"] = d.index.copy()
        d["exp"] = task.split("-")[-1] if not task.split("-")[-1].isnumeric() else "ass"
        data = d if data is None else pd.concat([data, d], axis=0)
    if reduce:
        circuits = [c.replace(".pickle", "").replace('.', '-')
                    for c in pickle.load(open("circuits-{}.pickle".format(exp), "rb"))]
        data = data.query("idx in @circuits")
    return data


def _plot_boxplot_on_ax(data, ax, y_label, title, x_labels):
    boxplot = ax.boxplot(data,
                         flierprops={"markerfacecolor": "gray"})
    for median in boxplot["medians"]:
        median.set_color("red")
    ax.set_xticks(list(range(1, len(data) + 1)),
                  x_labels,
                  fontsize=15)
    ax.set_ylabel(y_label,
                  fontsize=20)
    ax.set_title(title,
                 fontsize=20,
                 weight="bold")


def _plot_heatmap_on_ax(x, y, values, ax):
    ct = pd.crosstab(x, y,
                     values=values if values is not None else range(len(x)),
                     aggfunc="sum" if values is None else "mean",
                     margins=values is not None,
                     normalize="columns" if values is None else False)
    ct.replace(to_replace=0.0, value=np.nan, inplace=True)
    cmap = colormaps["viridis"]
    cmap.set_bad(color="dimgray")
    ax.imshow(ct.values, cmap=cmap)
    ax.set_xticks(np.arange(len(ct.columns)), labels=ct.columns, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(ct)), labels=ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = round(float(ct.values[i, j]), 2 if abs(ct.values[i, j]) < 1 else 0)
            ax.text(j, i, "N/A" if np.isnan(v) else str(v if v % 1 != 0 else int(v)),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=13)


def plot_figure_6(names={"single": "baseline", "es": "stimulation\nvalue", "ga": "# nodes\nstimulated", "rppo": "stimulation\ntime"},
                  exp="ass"):
    data = _load_all_data(reduce=True, exp=exp)
    algorithms = [algorithm for (algorithm,), _ in data.groupby(["algorithm"])]
    fig, axes = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    _plot_boxplot_on_ax([[np.mean([int(any(inner_d["is_broken"])) for _, inner_d in d.groupby(["idx"])])
                          for _, d in traj.groupby(["seed"])] for _, traj in data.groupby(["algorithm"])],
                        axes,
                        "% breakable memories",
                        "",
                        [names[a] for a in algorithms])
    axes.yaxis.set_major_formatter(
        tkr.FuncFormatter(lambda x, _: str(int(x * 100)))
    )
    print([np.std([np.mean([any(inner_d["is_broken"]) for _, inner_d in d.groupby(["idx"])])
                   for _, d in traj.groupby(["seed"])]) * 100
           for _, traj in data.groupby(["algorithm"])])
    axes.set_ylabel("% of breakable memories", fontsize=15)
    axes.set_ylim(0.0, 1.0)
    _flush_plot("figure_6.pdf")


def plot_figure_3(exp="ass"):
    data = _load_all_data(reduce=False, exp=exp)
    data = data[data["algorithm"] == "single"]
    data = data[data["exp"] == exp]
    data["network"] = data.apply(lambda row: row["idx"].split("-")[0], axis=1)
    ids = [i for (i,), traj in data.groupby(["network"])]
    fig, axes = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    axes.barh(np.arange(len(ids)),
              [np.mean([np.mean(d["is_broken"]) for _, d in traj.groupby(["idx"])])
               for _, traj in data.groupby(["network"])],
              color=COLORBREWER[1])
    axes.xaxis.set_major_formatter(
        tkr.FuncFormatter(lambda x, _: str(int(x * 100)))
    )
    axes.set_xlabel("% of breakable memories", fontsize=12)
    axes.set_ylabel("BioModels ID", fontsize=12)
    metadata = pd.read_csv("../GRNs/ontology.txt", sep=";")
    metadata.set_index("model_id", inplace=True)
    axes.set_yticks(np.arange(len(ids)), ["BIOMD" + ("0" * (10 - len(str(i)))) + str(i) for i in ids],
                    fontsize=7)
    plt.subplots_adjust(left=0.2)
    _flush_plot("figure_3.pdf")


def correlations_broken_w_property(alpha=0.05):
    networks = pd.read_csv("networks.txt", sep=";")
    ontologies = pd.read_csv("ontology.txt", sep=";")
    dynamics = pd.read_csv("dynamics.txt", sep=";")
    metadata = pd.merge(networks, ontologies, on="model_id", how="left")
    metadata = pd.merge(metadata, dynamics, on="model_id", how="left")
    memories = pd.read_csv("memories.txt", sep=";")
    metadata["is_broken"] = metadata.apply(
        lambda row: np.mean(memories[memories["model_id"] == row["model_id"]]["is_broken"]), axis=1)
    metadata["gene.ontology"] = metadata.apply(lambda row: row["gene.ontology"].split(",")[0], axis=1)
    metadata = metadata.dropna(subset=["is_broken"])
    metadata["lyapunov.mean"] = metadata.apply(
        lambda row: 1.0 if np.isinf(row["lyapunov.mean"]) else row["lyapunov.mean"], axis=1)
    corr = (metadata[metadata.columns.drop(["species.name", "species.common", "taxon", "homo", "gene.ontology"])]
            .corr(method="pearson"))
    print(corr)
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=3)
    cols = [col for col in metadata.columns.drop(["species.name",
                                                  "species.common",
                                                  "taxon",
                                                  "homo",
                                                  "gene.ontology",
                                                  "model_id",
                                                  "is_broken"])]
    for test, ax in zip([pearsonr, kendalltau, spearmanr], axes):
        corr = [(col, test(metadata[col], metadata["is_broken"])) for col in cols]
        print(corr)
        bars = ax.bar(np.arange(len(corr)), [c[1].statistic for c in corr])
        for i, c in enumerate(corr):
            if c[1].pvalue < alpha:
                bars[i].set_edgecolor("red")
                bars[i].set_linewidth(2)
        ax.set_xticks(np.arange(len(cols)), cols, rotation=90)
    _flush_plot("corr.png")


def plot_figure_7():
    networks = pd.read_csv("networks.txt", sep=";")
    ontologies = pd.read_csv("ontology.txt", sep=";")
    dynamics = pd.read_csv("dynamics.txt", sep=";")
    metadata = pd.merge(networks, ontologies, on="model_id", how="left")
    metadata = pd.merge(metadata, dynamics, on="model_id", how="left")
    memories = pd.read_csv("memories.txt", sep=";")
    metadata["is_broken"] = metadata.apply(
        lambda row: np.mean(memories[memories["model_id"] == row["model_id"]]["is_broken"]), axis=1)
    metadata["gene.ontology"] = metadata.apply(lambda row: row["gene.ontology"].split(",")[0], axis=1)
    metadata = metadata.dropna(subset=["is_broken"])
    metadata["lyapunov.mean"] = metadata.apply(
        lambda row: 1.0 if np.isinf(row["lyapunov.mean"]) else row["lyapunov.mean"], axis=1)

    fig, axes = plt.subplots(figsize=(20, 10), nrows=1, ncols=2, sharey=True)
    for var, ax in zip(["taxon", "gene.ontology"], axes):
        x = np.arange(len(metadata[var].unique()))
        heights = [np.mean(traj["is_broken"]) for _, traj in metadata.groupby([var])]
        print(var, heights)
        ax.bar(x,
               height=heights,
               capsize=5)
        ax.yaxis.set_major_formatter(
            tkr.FuncFormatter(lambda x, _: str(int(x * 100)))
        )
        ax.set_title("A) " + var if var == "taxon" else "B) " + var.replace(".", " "), weight="bold", fontsize=25)
        labels = [i for (i,), _ in metadata.groupby([var])]
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=15, rotation=25)
        ax.tick_params(axis="y", labelsize=15)
    axes[0].set_ylabel("% breakable memories", fontsize=20)
    fig.tight_layout()
    _flush_plot("figure_7.pdf")


def plot_figure_8():
    data = pd.read_csv("info.txt", sep=";")
    print("ALPHA: {}".format(0.05 / len(data)))
    for row, measure in enumerate(["emergence"]):
        print("==========={}===========".format(measure))
        _compute_paired_cols(data=data, measure=measure)
        res = wilcoxon(data["test->reset"], alternative="greater")
        print("test->reset: " + str(res.pvalue))
    fig, axes = plt.subplots(figsize=(16, 5), nrows=1, ncols=2)
    boxes = []
    for ax, label in zip(axes, ["stimuli that reset", "stimuli that don't reset"]):
        box = filter_outliers(data[data["is_mem"] != bool("don't" in label)]["test->reset"])
        boxplot = ax.boxplot(filter_outliers(data[data["is_mem"] != bool("don't" in label)]["test->reset"]))
        boxes.append(box)
        for median in boxplot["medians"]:
            median.set_color("red")
        ax.set_ylim(-10, 20)
        ax.hlines(0.0, 0.5, 1.5, color="black", alpha=0.5, linestyles="dashed")
        ax.set_xlabel(label, fontsize=15)
        if "don't" not in label:
            ax.set_ylabel("% change in integrated information", fontsize=15)
        else:
            ax.set_yticks([])
        ax.set_xticks([])
    print(mannwhitneyu(boxes[0], boxes[1], alternative="greater"))
    fig.tight_layout()
    _flush_plot("figure_8.pdf")


def filter_outliers(data):
    q25, q75 = data.quantile(0.25), data.quantile(0.75)
    iqr_threshold = (q75 - q25) * 1.5
    return data[(q25 - iqr_threshold <= data) & (data <= q75 + iqr_threshold)]


def _compute_paired_cols(data, measure):
    for i, p in enumerate(PHASES):
        first_col = ".".join([p, measure])
        for j, other_p in enumerate(PHASES):
            second_col = ".".join([other_p, measure])
            if first_col in data.columns and second_col in data.columns and i < j:
                data["->".join([p, other_p])] = ((data[second_col] - data[
                    first_col]) / (data[first_col] + 1e-10)) * 100


def uniform_median_beta_test(x, a, b, alternative="greater"):
    """
    Beta/order-statistic test for the sample median under H0: X ~ Uniform(a, b),
    where a and b are KNOWN.

    For odd n=2k+1, the sample median is X_(k+1). Under H0:
        U = (X_(k+1) - a)/(b-a) ~ Beta(k+1, k+1)

    For even n, this function uses the LOWER middle order statistic X_(n/2)
    as the "median" for testing, for which:
        U = (X_(n/2) - a)/(b-a) ~ Beta(n/2, n/2 + 1)

    Parameters
    ----------
    x : array-like
        Sample data.
    a, b : float
        Known Uniform bounds with a < b.
    alternative : {"greater","less","two-sided"}
        Hypothesis about the median location relative to the uniform null.
        - "greater": median is higher than expected under Uniform(a,b)
        - "less": median is lower
        - "two-sided": median differs

    Returns
    -------
    result : dict
        {
          "n": sample size,
          "median_def": description of which order statistic was used,
          "m_obs": observed median statistic used,
          "u_obs": scaled statistic in [0,1],
          "beta_shape": (alpha, beta),
          "pvalue": p-value,
        }

    Notes
    -----
    - Requires all x within [a,b] for the strict Uniform(a,b) null.
      If not, the null is already violated; we raise ValueError.
    - If you prefer the UPPER middle for even n, see `uniform_median_beta_test_upper`.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array.")
    if not (a < b):
        raise ValueError("Require a < b.")
    if np.any(x < a) or np.any(x > b):
        raise ValueError("Some observations fall outside [a,b]; Uniform(a,b) null is violated.")

    n = x.size
    xs = np.sort(x)

    if n % 2 == 1:
        # odd: true sample median is X_(k+1)
        k = n // 2  # 0-indexed middle position
        m_obs = xs[k]
        alpha = beta_param = k + 1
        beta_param = k + 1
        median_def = f"odd n: order statistic X_({k+1}) (sample median)"
    else:
        # even: choose LOWER middle order statistic X_(n/2)
        k = n // 2  # e.g., n=10 -> k=5; lower middle is X_(5) in 1-indexing
        m_obs = xs[k - 1]
        alpha = k
        beta_param = k + 1
        median_def = f"even n: LOWER middle order statistic X_({k})"

    # scale to U in [0,1]
    u_obs = (m_obs - a) / (b - a)
    # numerical guard
    u_obs = min(max(u_obs, 0.0), 1.0)

    # compute p-value under Beta(alpha, beta_param)
    cdf = beta.cdf(u_obs, alpha, beta_param)

    alternative = alternative.lower()
    if alternative == "greater":
        p = 1.0 - cdf
    elif alternative == "less":
        p = cdf
    elif alternative in ("two-sided", "two_sided", "two sided"):
        p = 2.0 * min(cdf, 1.0 - cdf)
        p = min(p, 1.0)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    return {
        "n": n,
        "median_def": median_def,
        "m_obs": float(m_obs),
        "u_obs": float(u_obs),
        "beta_shape": (int(alpha), int(beta_param)),
        "pvalue": float(p),
    }


def uniform_median_beta_test_upper(x, a, b, alternative="greater"):
    """
    Same as uniform_median_beta_test, but for even n uses the UPPER middle
    order statistic X_(n/2 + 1), where under H0:
        U ~ Beta(n/2 + 1, n/2)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D array.")
    if not (a < b):
        raise ValueError("Require a < b.")
    if np.any(x < a) or np.any(x > b):
        raise ValueError("Some observations fall outside [a,b]; Uniform(a,b) null is violated.")

    n = x.size
    xs = np.sort(x)

    if n % 2 == 1:
        return uniform_median_beta_test(x, a, b, alternative=alternative)

    k = n // 2  # n=10 -> k=5; upper middle is X_(6)
    m_obs = xs[k]
    alpha = k + 1
    beta_param = k
    u_obs = (m_obs - a) / (b - a)
    u_obs = min(max(u_obs, 0.0), 1.0)

    cdf = beta.cdf(u_obs, alpha, beta_param)

    alternative = alternative.lower()
    if alternative == "greater":
        p = 1.0 - cdf
    elif alternative == "less":
        p = cdf
    elif alternative in ("two-sided", "two_sided", "two sided"):
        p = 2.0 * min(cdf, 1.0 - cdf)
        p = min(p, 1.0)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    return {
        "n": n,
        "median_def": f"even n: UPPER middle order statistic X_({k+1})",
        "m_obs": float(m_obs),
        "u_obs": float(u_obs),
        "beta_shape": (int(alpha), int(beta_param)),
        "pvalue": float(p),
    }


def plot_figure_9():
    optima = pd.read_csv("optima.txt", sep=";")
    optima = optima.replace([np.inf, -np.inf], np.nan).fillna(optima.median())
    networks = pd.read_csv("networks.txt", sep=";")
    dynamics = pd.read_csv("dynamics.txt", sep=";")
    bio = pd.merge(networks, dynamics, on=["model_id"])
    print(spearmanr(optima["memories.perc"], optima["emergence.median"]),
          pearsonr(optima["memories.perc"], optima["emergence.median"]),
          kendalltau(optima["memories.perc"], optima["emergence.median"]))

    optima["is_bio"] = False
    bio["is_bio"] = True
    bio["is_best_ce"] = False
    bio["is_best_mem"] = False
    both_df = pd.concat([optima[bio.columns], bio], axis=0)
    cols = NETWORK_PROPERTIES + DYNAMICAL_PROPERTIES
    both_df[cols] = both_df[cols].replace([np.inf, -np.inf], np.nan).fillna(both_df[cols].median())
    both_df[cols] = (both_df[cols] - both_df[cols].mean()) / both_df[cols].std()
    both_df[cols] = both_df[cols].replace([np.inf, -np.inf], np.nan).fillna(both_df[cols].median())
    optimal_ce = both_df[both_df["is_best_ce"] == True][cols].values.ravel()
    optimal_learning = both_df[both_df["is_best_mem"] == True][cols].values.ravel()
    both_df = both_df[both_df["is_best_ce"] == False]
    both_df = both_df[both_df["is_best_mem"] == False]
    original_cols = both_df[cols].copy()
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)
    for ax, target in zip(axes, [optimal_learning, optimal_ce]):
        both_df[cols] = original_cols.copy()
        for i, col in enumerate(cols):
            both_df[col] = both_df.apply(lambda row: np.linalg.norm(row[col] - target[i]), axis=1)
        both_df[cols] = both_df[cols].replace([np.inf, -np.inf], np.nan).fillna(both_df[cols].median())
        bio_vals, random_vals = (both_df[both_df["is_bio"] == True][cols].values.ravel(),
                                 both_df[both_df["is_bio"] == False][cols].values.ravel())

        print(mannwhitneyu(random_vals, bio_vals, alternative="less"))
        print(len(random_vals), len(bio_vals))
        print(np.mean([x < y for x in random_vals for y in bio_vals]))
        print(uniform_median_beta_test(random_vals, np.nanmin(random_vals), np.nanmax(random_vals), alternative="less"))
        print(uniform_median_beta_test(bio_vals, np.nanmin(bio_vals), np.nanmax(bio_vals), alternative="less"))

        ax.vlines([np.nanmedian(bio_vals)], ymin=0, ymax=250, color="blue", label="biological median\n(real)")
        ax.vlines([np.nanmedian(random_vals)], ymin=0, ymax=250, color="red", label="synthetic median\n(real)")
        ax.vlines([(np.nanmax(bio_vals) - np.nanmin(bio_vals)) / 2.0], ymin=0, ymax=250, color="blue", linestyles="dashed", label="biological median\n(hypothetical uniform)")
        ax.vlines([(np.nanmax(random_vals) - np.nanmin(random_vals)) / 2.0], ymin=0, ymax=250, color="red", linestyles="dashed", label="synthetic median\n(hypothetical uniform)")
        ax.hist(bio_vals, bins=10, color="blue", alpha=0.5)
        ax.hist(random_vals, bins=10, color="red", alpha=0.5)
        ax.set_xlabel("distance from optimum\n(standardized)")
        ax.set_ylabel("count")
        ax.set_title("Optimally learning network" if target is optimal_learning else "Optimally causal emergent network")
        ax.legend()

    fig.tight_layout()
    _flush_plot("figure_9.pdf")


if __name__ == "__main__":
    plot_figure_3()
    plot_figure_6()
    plot_figure_7()
    # correlations_broken_w_property()
    plot_figure_8()
    plot_figure_9()
