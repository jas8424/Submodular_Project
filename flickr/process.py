import sys
import os
import pickle
import collections
import random
from numpy import sort
from scipy import rand
id = 0
users = dict()
os.chdir(sys.path[0])

with open("release-flickr-groupmemberships.txt", "r") as f:
    lines = f.readlines()
    cnt = collections.defaultdict(int)
    for line in lines:
        u, v = map(int, line.strip().split())
        cnt[v] += 1
    # print(cnt)
    l = [(cnt[key], key) for key in cnt]
    # print(l)
    l = sorted(l, reverse=True)
    l = l[:2000]
    random.shuffle(l)
    id = 0
    idx = dict()
    items = dict()
    st = set()
    for x in l:
        idx[x[1]] = id
        st.add(x[1])
        items[id] = id
        id += 1
    uidx = dict()
    uid = 0
    for line in lines:
        u, v = map(int, line.strip().split())
        if v in idx:
            if u not in uidx:
                uidx[u] = uid
                uid += 1
    users = dict((i, set()) for i in range(uid))
    for line in lines:
        u, v = map(int, line.strip().split())
        if u not in uidx or v not in idx:
            continue
        users[uidx[u]].add(idx[v])

with open("items.pkl", "wb") as f:
    pickle.dump(items, f)

with open("users.pkl", "wb") as f:
    pickle.dump(users, f)

# print(users)
