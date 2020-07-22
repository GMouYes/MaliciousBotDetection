import pickle
from pprint import pprint
with open("hyper.pkl","rb") as f:
	pprint(pickle.load(f))
