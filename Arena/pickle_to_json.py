

import pickle
from json import  dump


with open(f'misharon_policy', 'rb') as myfile:
    _, my_policy, _, _ = pickle.load(myfile)

# https://stackoverflow.com/questions/7001606/json-serialize-a-dictionary-with-tuples-as-key

# save: convert each tuple key to a string before saving as json object
with open('misharon_policy.json', 'w') as myfile:
    to_save = {str(k): int(v) for k, v in my_policy.items()}
    dump(to_save, myfile)

pass