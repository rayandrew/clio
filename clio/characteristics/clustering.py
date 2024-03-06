import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import Annotated

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import pairwise_distance

import typer

from clio.utils.characteristic import Characteristic, Characteristics, Statistic
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.query import QueryExecutionException, get_query
from clio.utils.trace_pd import TraceWindowGeneratorContext, read_dataset_as_df, read_labeled_as_df, trace_time_window_generator

app = typer.Typer(name="Characteristic Clustering", pretty_exceptions_enable=False)


def preliminary_process(output: Path, log_level: LogLevel, dir: Path, query: str):
    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Run clustering on %s", dir, tab=0)

    is_dataset = True
    data_paths = [p for p in dir.glob("**/*.dataset")]
    if len(data_paths) == 0:
        data_paths = [p for p in dir.glob("**/*.labeled")]
        is_dataset = False

    if len(data_paths) == 0:
        log.error("No datasets found in %s", dir)
        sys.exit(1)

    reader = read_dataset_as_df if is_dataset else read_labeled_as_df

    # TODO: remove this
    # for debugging purposes, only use the first dataset
    # data_paths = [data_paths[0]]

    try:
        q = get_query(query)
        data = pd.concat([reader(path) for path in data_paths])
        if q:
            data: pd.DataFrame = data[q({"data": data})]  # type: ignore
        data = data.tail(65777)
    except QueryExecutionException as e:
        log.error("Failed to execute expression: %s", e)
        sys.exit(1)

    scaler = StandardScaler()
    # remove latency and reject if any
    data = data.drop(columns=["latency", "reject"], errors="ignore")
    data = data.dropna()
    log.info("Data columns: %s", list(data.columns), tab=1)
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return log, data_scaled


@app.command()
def dbscan(
    dir: Annotated[Path, typer.Argument(help="The characteristics file to plot", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    from sklearn.cluster import DBSCAN

    log, ori_data = preliminary_process(output, log_level, dir, query)

    # dbscan = DBSCAN(eps=0.1, min_samples=500, n_jobs=-1)
    # dbscan = DBSCAN(eps=0.1, min_samples=100, n_jobs=-1)
    dbscan = DBSCAN(eps=0.1, min_samples=30, n_jobs=-1)
    start_time = timer()
    data = ori_data.copy()
    dbscan = dbscan.fit(data)
    end_time = timer()
    log.info("Time to fit: %f", end_time - start_time, tab=1)

    labels = set(dbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # type: ignore
    log.info("Number of clusters: %d", n_clusters, tab=1)
    n_noise_ = list(labels).count(-1)  # type: ignore
    log.info("Number of noise points: %d", n_noise_, tab=1)

    # predict each data and save cluster
    data["cluster"] = dbscan.labels_
    data.to_csv(output / "data.csv", index=False)

    # save clusters centers
    centroids = {}
    for label in labels:
        if label == -1:
            continue
        cluster = data[data["cluster"] == label]

        cluster = cluster.drop(columns=["cluster"])
        centroid = np.mean(cluster, axis=0)
        centroids[label] = centroid
    centroids = pd.DataFrame(centroids).T
    centroids.to_csv(output / "centroids.csv", index=False)

    # plot clusters and data using PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    start_time = timer()
    data_pca = pca.fit_transform(ori_data)
    end_time = timer()
    log.info("Time to PCA transform: %f", end_time - start_time, tab=1)
    data_pca = pd.DataFrame(data_pca, columns=["x", "y"])  # type: ignore
    data_pca["cluster"] = dbscan.labels_

    fig, ax = plt.subplots(figsize=(10, 10))
    for label in labels:
        if label == -1:
            continue
        cluster = data_pca[data_pca["cluster"] == label]
        ax.scatter(cluster["x"], cluster["y"], label=label)
        centroid = np.mean(cluster, axis=0)
        ax.scatter(centroid["x"], centroid["y"], label=label, color="black")
    ax.legend()
    plt.savefig(output / "clusters_pca.png")
    plt.close(fig)

    # plot clusters and data using TSNE
    # from MulticoreTSNE import MulticoreTSNE as TSNE
    from fitsne import FItSNE as fitsne

    # tsne = TSNE(n_components=2, n_jobs=4)

    start_time = timer()
    # data_tsne = tsne.fit_transform(ori_data)
    data_tsne = fitsne(np.ascontiguousarray(ori_data.values), nthreads=8)
    end_time = timer()

    log.info("Time to TSNE transform: %f", end_time - start_time, tab=1)
    data_tsne = pd.DataFrame(data_tsne, columns=["x", "y"])  # type: ignore
    data_tsne["cluster"] = dbscan.labels_

    # find centroids in TSNE, do not use fit_transform
    centroids = {}
    for label in labels:
        if label == -1:
            continue
        cluster = data_tsne[data_tsne["cluster"] == label]
        centroid = np.mean(cluster, axis=0)
        centroids[label] = centroid
    centroids = pd.DataFrame(centroids).T
    centroids.to_csv(output / "centroids_tsne.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    for label in labels:
        if label == -1:
            continue
        cluster = data_tsne[data_tsne["cluster"] == label]
        ax.scatter(cluster["x"], cluster["y"], label=label)
        centroid = centroids.loc[label]
        ax.scatter(centroid["x"], centroid["y"], label=label, color="black")
    # ax.legend()
    plt.savefig(output / "clusters_tsne.png")
    plt.close(fig)


@app.command()
def hdbscan(
    dir: Annotated[Path, typer.Argument(help="The characteristics file to plot", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    from sklearn.cluster import HDBSCAN

    log, data = preliminary_process(output, log_level, dir, query)

    hdbscan = HDBSCAN(min_cluster_size=15, min_samples=10)
    hdbscan = hdbscan.fit(data)

    labels = set(hdbscan.labels_)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # type: ignore
    log.info("Number of clusters: %d", n_clusters, tab=1)
    n_noise_ = list(labels).count(-1)  # type: ignore
    log.info("Number of noise points: %d", n_noise_, tab=1)


@app.command()
def kmeans(
    dir: Annotated[Path, typer.Argument(help="The characteristics file to plot", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    # from sklearn.cluster import KMeans
    import torch

    from fast_pytorch_kmeans import KMeans

    log, data = preliminary_process(output, log_level, dir, query)
    data_tensor = torch.tensor(data.values, device="cuda")

    # Pick best number of clusters using elbow method and silhouette score
    start_time = timer()
    results = {}
    for n_clusters in range(2, 20):
        log.info("Number of clusters: %d", n_clusters, tab=1)
        kmeans = KMeans(n_clusters=n_clusters)
        pred = kmeans.fit_predict(data_tensor)
        centroids = kmeans.centroids  # type: ignore
        inertia = torch.sum((data_tensor.unsqueeze(1) - centroids.unsqueeze(0)) ** 2, dim=2).min(dim=1).values.sum()  # type: ignore
        log.info("Inertia: %f", inertia, tab=1)
        score = silhouette_score(data.values, pred.cpu().numpy())  # type: ignore
        # NOTE: using sklearn
        # kmeans = KMeans(n_clusters=n_clusters)
        # kmeans.fit(data)
        # log.info("Inertia: %f", kmeans.inertia_, tab=1)
        # score = silhouette_score(data, kmeans.labels_)
        # results[n_clusters] = {"inertia": kmeans.inertia_, "silhouette": score}
        log.info("Silhouette score: %f", score, tab=1)
        results[n_clusters] = {"inertia": inertia.cpu().item(), "silhouette": score.item()}
    end_time = timer()

    results = pd.DataFrame(results).T
    results.to_csv(output / "results.csv")

    log.info("Time to do elbow method: %f", end_time - start_time, tab=1)
    best_n_clusters_elbow = results.idxmin()["inertia"]
    best_n_clusters_silhouette = results.idxmax()["silhouette"]
    log.info("Best number of clusters based on elbow: %d", best_n_clusters_elbow, tab=1)
    log.info("Best number of clusters based on silhouette: %d", best_n_clusters_silhouette, tab=1)

    # plot the inertia and silhouette score
    fig, ax = plt.subplots()
    ax.plot(results.index, results["inertia"], label="Inertia")
    # ax.plot(results.keys(), [results[x]["inertia"] for x in results], label="Inertia")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    # format the ticks to be integers and multiples of 2
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2 = ax.twinx()
    ax2.plot(results.index, results["silhouette"], label="Silhouette", color="orange")
    # ax2.plot(results.keys(), [results[x]["silhouette"] for x in results], label="Silhouette", color="orange")
    # add legend
    ax.legend(loc="upper left")
    ax2.set_ylabel("Silhouette score")
    fig.tight_layout()
    plt.savefig(output / "elbow.png")
    plt.close(fig)


if __name__ == "__main__":
    app()
