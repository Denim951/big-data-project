import os
import csv
import random
from math import sqrt

FILE_NAME = "data/raw/ratings.csv"
MOVIES_NAME = "data/raw/movies.csv"
MIN_COMMON = 5


def load_ratings(path):
    users = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user = row["userId"]
            if user not in users:
                users[user] = {}
            users[user][row["movieId"]] = float(row["rating"])
    return users

def load_titles(path):
    titles = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles[row["movieId"]] = row["title"]
    return titles


def average(ratings):
    return sum(ratings.values()) / len(ratings)


def pearson(a, b, min_common=MIN_COMMON):
    common = [item for item in a if item in b]
    if len(common) < min_common:
        return 0.0
    avg_a = average(a)
    avg_b = average(b)
    top = 0.0
    left = 0.0
    right = 0.0
    for item in common:
        da = a[item] - avg_a
        db = b[item] - avg_b
        top += da * db
        left += da * da
        right += db * db
    if left == 0 or right == 0:
        return 0.0
    return top / sqrt(left * right)


def neighbors(users, user, min_common=MIN_COMMON):
    sims = []
    for other in users:
        if other != user:
            s = pearson(users[user], users[other], min_common)
            if s > 0:
                sims.append((s, other))
    sims.sort(reverse=True)
    return sims


def predict(users, user, item, k, adjust=False, min_common=MIN_COMMON):
    chosen = []
    for s, other in neighbors(users, user, min_common):
        if item in users[other]:
            chosen.append((s, other))
        if len(chosen) == k:
            break
    if len(chosen) == 0:
        return average(users[user])
    top = 0.0
    bottom = 0.0
    for s, other in chosen:
        if adjust:
            top += s * (users[other][item] - average(users[other]))
        else:
            top += s * users[other][item]
        bottom += s
    if adjust:
        return average(users[user]) + top / bottom
    return top / bottom


def recommend(users, user, k, how_many=5):
    scores = {}
    for s, other in neighbors(users, user)[:k]:
        for item, rating in users[other].items():
            if item not in users[user]:
                top, bottom, count = scores.get(item, (0.0, 0.0, 0))
                scores[item] = (top + s * rating, bottom + s, count + 1)
    results = []
    for item, (top, bottom, count) in scores.items():
        if count >= 3:
            results.append((top / bottom, item))
    results.sort(reverse=True)
    return results[:how_many]


def small_table():
    movies = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
    table = {
        "U1":  [5, 4, None, 1, 2, 5, None, 1, 4, 2],
        "U2":  [4, 5, 4, 2, None, 4, 1, 2, 5, 1],
        "U3":  [1, 2, 1, 5, 4, None, 5, 4, 1, 5],
        "U4":  [5, 5, 3, None, 1, 5, 2, 1, 4, 2],
        "U5":  [2, 1, 2, 4, 5, 2, 4, None, 2, 4],
        "U6":  [None, 4, 5, 1, 2, 4, 1, 2, 5, 1],
        "U7":  [1, 1, 2, 5, 5, 1, 5, 4, 2, None],
        "U8":  [4, 4, 4, 2, 1, None, 2, 1, 4, 2],
        "U9":  [2, 2, 1, 4, 4, 2, 5, 5, None, 4],
        "U10": [5, 4, 5, 1, None, 4, 1, 1, 5, 2],
    }
    users = {}
    for name, row in table.items():
        users[name] = {}
        for movie, rating in zip(movies, row):
            if rating is not None:
                users[name][movie] = float(rating)
    return users


def pick_test_ratings(users, how_many):
    random.seed(1)
    tests = []
    for user in random.sample(list(users), how_many):
        item = random.choice(list(users[user]))
        tests.append((user, item))
    return tests


def rmse(users, tests, k, method):
    total = 0.0
    for user, item in tests:
        real = users[user].pop(item)
        if method == "user average":
            guess = average(users[user])
        elif method == "weighted average":
            guess = predict(users, user, item, k)
        else:
            guess = predict(users, user, item, k, adjust=True)
        users[user][item] = real
        total += (guess - real) ** 2
    return sqrt(total / len(tests))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)

    print("#####Testing on 10 users and 10 movies#####")
    small = small_table()
    print("Similarities with U1:")
    for other in small:
        if other != "U1":
            print("  U1 and", other, ":", round(pearson(small["U1"], small[other], 3), 2))
    for movie in ["M3", "M7"]:
        guess = predict(small, "U1", movie, k=3, min_common=3)
        print("U1 has not rated", movie, "- predicted rating:", round(guess, 1))

    print("\n#####Testing on MovieLens data#####")
    users = load_ratings(os.path.join(project, FILE_NAME))
    titles = load_titles(os.path.join(project, MOVIES_NAME))
    print("Users:", len(users))
    print("Movies:", len({item for ratings in users.values() for item in ratings}))
    print("Ratings:", sum(len(r) for r in users.values()))

    tests = pick_test_ratings(users, 500)
    print("\nRMSE baseline:", round(rmse(users, tests, 0, "user average"), 3))
    for k in [10, 20, 50]:
        plain = rmse(users, tests, k, "weighted average")
        adjusted = rmse(users, tests, k, "adjusted")
        print("k =", k, "| plain:", round(plain, 3), "| adjusted:", round(adjusted, 3))

    print("\nTop 5 picks for MovieLens user 1:")
    for score, item in recommend(users, "1", 20):
        print("  ", titles.get(item, item), "| predicted", round(score, 1))