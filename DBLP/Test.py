import pickle

with open("graph.pkl","rb") as f:
    data=pickle.load(f)

print(data)