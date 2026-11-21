import os
from itertools import combinations
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FILE_NAME = "data/raw/spotify-huge-audio-features.parquet"

COLUMNS = ["track_id", "track_popularity", "explicit", "duration_ms", "tempo",
           "mode", "danceability", "energy", "loudness", "speechiness",
           "acousticness", "instrumentalness", "liveness", "valence"]

FEATURES = ["danceability", "energy", "valence", "acousticness",
            "speechiness", "liveness", "loudness", "tempo"]


def clean(df):
    df = df.dropna()
    df = df[(df["tempo"] > 0) & (df["duration_ms"] > 0)]
    return df.drop_duplicates("track_id")


def find_cutoffs(file):
    pieces = []
    for i in range(file.num_row_groups):
        chunk = file.read_row_group(i, columns=COLUMNS).to_pandas()
        pieces.append(chunk.sample(frac=0.01, random_state=1))
    sample = clean(pd.concat(pieces))

    cutoffs = {}
    for f in FEATURES:
        cutoffs[f] = (sample[f].quantile(1 / 3), sample[f].quantile(2 / 3))
    cutoffs["popular"] = sample["track_popularity"].quantile(0.9)
    return cutoffs


def trait_table(df, cutoffs):
    traits = pd.DataFrame(index=df.index)
    for f in FEATURES:
        low_cut, high_cut = cutoffs[f]
        high = df[f] >= high_cut
        low = (df[f] <= low_cut) & ~high
        traits["high_" + f] = high
        traits["low_" + f] = low
    traits["popular"] = df["track_popularity"] >= cutoffs["popular"]
    traits["explicit"] = df["explicit"].astype(bool)
    traits["instrumental"] = df["instrumentalness"] > 0.5
    traits["major_key"] = df["mode"] == 1
    traits["minor_key"] = df["mode"] != 1
    return traits


def count_song_types(file, cutoffs):
    total = {}
    for i in range(file.num_row_groups):
        chunk = clean(file.read_row_group(i, columns=COLUMNS).to_pandas())
        traits = trait_table(chunk, cutoffs)
        names = list(traits.columns)
        powers = 2 ** np.arange(len(names))
        codes = (traits.to_numpy() * powers).sum(axis=1)
        for code, count in pd.Series(codes).value_counts().items():
            total[int(code)] = total.get(int(code), 0) + int(count)
        if i % 10 == 0:
            print("  chunk", i + 1, "of", file.num_row_groups)
    return total, names


def make_baskets(total, names):
    baskets = []
    weights = []
    for code, count in total.items():
        basket = set()
        for i in range(len(names)):
            if (code >> i) & 1:
                basket.add(names[i])
        baskets.append(basket)
        weights.append(count)
    return baskets, weights


def apriori(baskets, weights, s):
    frequent = {}

    counts = {}
    for basket, w in zip(baskets, weights):
        for item in basket:
            counts[(item,)] = counts.get((item,), 0) + w
    current = {}
    for group in counts:
        if counts[group] >= s:
            current[group] = counts[group]
    frequent.update(current)
    print("  pass 1:", len(current), "common traits")

    k = 2
    while len(current) > 0:
        common_items = []
        for group in current:
            for item in group:
                if item not in common_items:
                    common_items.append(item)
        common_items.sort()

        candidates = set()
        for group in current:
            for item in common_items:
                if item > group[-1]:
                    new_group = group + (item,)
                    keep = True
                    for i in range(len(new_group)):
                        piece = new_group[:i] + new_group[i + 1:]
                        if piece not in current:
                            keep = False
                    if keep:
                        candidates.add(new_group)

        counts = {}
        for basket, w in zip(baskets, weights):
            items = sorted(i for i in basket if i in common_items)
            for group in combinations(items, k):
                if group in candidates:
                    counts[group] = counts.get(group, 0) + w

        current = {}
        for group in counts:
            if counts[group] >= s:
                current[group] = counts[group]
        frequent.update(current)
        print("  pass", k, ":", len(candidates), "checked,", len(current), "kept")
        k += 1

    return frequent


def make_rules(frequent, n, c):
    rules = []
    for group in frequent:
        if len(group) < 2:
            continue
        for j in group:
            left = tuple(i for i in group if i != j)
            confidence = frequent[group] / frequent[left]
            if confidence >= c:
                interest = confidence - frequent[(j,)] / n
                rules.append((confidence, interest, left, j))
    rules.sort(reverse=True)
    return rules


def show_rules(rules, top=10):
    for confidence, interest, left, j in rules[:top]:
        print("  ", ", ".join(left), "->", j,
              "| right", round(confidence * 100), "% of the time")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)
    file = pq.ParquetFile(os.path.join(project, FILE_NAME))

    print("Finding cutoffs")
    cutoffs = find_cutoffs(file)
    print("Completed.")

    print("Reading the whole file")
    total, names = count_song_types(file, cutoffs)
    baskets, weights = make_baskets(total, names)

    n = sum(weights)
    popular_share = sum(w for b, w in zip(baskets, weights) if "popular" in b) / n
    print("songs used:", n)
    print("different trait combinations:", len(baskets))
    print("share of songs that are popular:", round(popular_share * 100, 1), "%")

    print("\n##### SMALL CHECK (10 songs) #####")
    small = baskets[::len(baskets) // 10][:10]
    print("items:", len(set().union(*small)))
    frequent = apriori(small, [1] * 10, 3)
    show_rules(make_rules(frequent, 10, 0.7))

    print("\n##### ALL SONGS #####")
    for percent in [10, 5, 3]:
        s = int(n * percent / 100)
        print("\nthreshold:", percent, "% of songs")
        frequent = apriori(baskets, weights, s)
        print("  common groups found:", len(frequent))
        for c in [0.9, 0.7, 0.5]:
            print("  confidence", c, "->", len(make_rules(frequent, n, c)), "rules")

    print("\nStrongest rules (threshold 5%, confidence 70%+):")
    frequent = apriori(baskets, weights, int(n * 0.05))
    show_rules(make_rules(frequent, n, 0.7), top=10)

    print("\n##### RULES THAT END IN 'popular' (threshold 1%) #####")
    print("baseline: popular songs are", round(popular_share * 100, 1), "% of all songs")
    frequent = apriori(baskets, weights, int(n * 0.01))
    found = []
    for confidence, interest, left, j in make_rules(frequent, n, 0):
        if j == "popular":
            found.append((interest, confidence, left))
    found.sort(reverse=True)
    for interest, confidence, left in found[:10]:
        print("  ", ", ".join(left), "-> popular | popular",
              round(confidence * 100), "% of the time")
    if len(found) == 0:
        print("  none found")