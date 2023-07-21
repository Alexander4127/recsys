import spark
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_spark(df, recs, relevance, k):
    recs = recs.reshape(-1)
    repeated_users = np.repeat(df["user_idx"], k)
    df = df.merge(repeated_users, on="user_idx")
    assert len(df) == len(recs), f'{len(df)} != {len(recs)}'
    df["item_idx"] = recs
    df["relevance"] = relevance.reshape(-1)
    return spark.createDataFrame(df)


def arm_selection(sim, users):
    item_recs = sim.get_log(users).groupBy(["__iter", "item_idx"]).count().toPandas().rename(columns={"count": "item_count"})
    all_recs = sim.get_log(users).groupBy(["__iter"]).count().toPandas().rename(columns={"count": "all_count"})[["__iter", "all_count"]]
    recs = pd.merge(item_recs, all_recs).sort_values(by="__iter")
    recs["item_count"] /= recs["all_count"]
    return recs


def cum_regret(sim, users):
    success_count = sim.get_log(users).orderBy("__iter").groupBy(["__iter"]).sum().toPandas()["sum(response)"]
    all_count = sim.get_log(users).orderBy("__iter").groupBy(["__iter"]).count().toPandas()["count"]
    return all_count - success_count


def plot_selection(recommendations, names, w=15, h=5):
    plt.gcf().set_size_inches(w, h)
    plt.suptitle("Arm selection plot")
    for index, model_name in enumerate(names):
        plt.subplot(1, len(names), index + 1)
        recs = recommendations[index]
        for item_idx in np.unique(recs["item_idx"]):
            cur_recs = recs[recs["item_idx"] == item_idx]
            plt.plot(cur_recs["__iter"], cur_recs["item_count"], label=f'Bandit {item_idx}')

        plt.title(f"Arm selection for {model_name}", fontsize=15)
        plt.xlabel("iteration", fontsize=12)
        plt.ylabel("dist between arms", fontsize=12)
        plt.legend()
    plt.show()
