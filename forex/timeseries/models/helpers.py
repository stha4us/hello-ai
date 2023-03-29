import pickle

def read_pickle(data_path):
    with open(data_path, 'rb+') as file:
        content = pickle.load(file)
        return content