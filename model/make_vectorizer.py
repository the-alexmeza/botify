import pickle

from preprocess import make_vectorizer

if __name__ == '__main__':
    with open('vectorizer_dict.kai', 'rb') as dict_file:
        make_vectorizer(pickle.load(dict_file))
