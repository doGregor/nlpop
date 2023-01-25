import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from flair.embeddings import TransformerDocumentEmbeddings, WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence


def read_data(file='final'):
    file_path = '../cache/' + file + '.csv'
    df = pd.read_csv(file_path, sep=',')
    return df


def split_row_text(text):
    return text.split('<THREAD_SEP>')


def split_tweet_ids(tweet_ids):
    return tweet_ids[1:-1].replace("'", '').replace('\n', '').split(' ')


def list_string_to_list(list_string):
    list_string = ast.literal_eval(list_string)
    return list_string


def numeric_list_string_to_list(numeric_list_string):
    numeric_list_string = numeric_list_string[1:-1].split()
    return [int(x) for x in numeric_list_string]


def sep_arxive_data(arxive_string):
    arxive_string = arxive_string.split('<THREAD_SEP>')
    arxive_string = '<ARXIV_SEP>'.join(arxive_string)
    arxive_string = arxive_string.split('<ARXIV_SEP>')
    return arxive_string


def text_features_to_numeric(text_array, text_type='tweet'):
    if text_type not in ['tweet', 'paper']:
        raise Exception(f'text_type must be either "tweet" or "paper" but was {text_type}')
    if text_type == 'tweet':
        EMBEDDING = TransformerDocumentEmbeddings('vinai/bertweet-base')
    else:
        EMBEDDING = TransformerDocumentEmbeddings('bert-base-uncased')
    embedding_array = []
    for sentence in tqdm(text_array):
        if sentence.strip() == '':
            embedding_array.append(np.zeros(768))
        else:
            sent = Sentence(sentence)
            EMBEDDING.embed(sent)
            embedding_array.append(sent.embedding.cpu().detach().numpy())
    return np.asarray(embedding_array)


if __name__ == '__main__':
    '''
    df = read_data('valid')

    text_col = df['text'].to_list()
    text_col = [text.split('<THREAD_SEP>') for text in text_col]

    tweet_ids = df['tweet_id'].to_list()
    tweet_ids = [ids[1:-1].replace("'", '').replace('\n', '').split(' ') for ids in tweet_ids]

    for idx, val in enumerate(text_col):
        if len(val) != len(tweet_ids[idx]):
            print(idx)
    '''

    my_authors = 'test1\ttest2<THREAD_SEP>test1\ttest2<ARXIV_SEP>test1\ttest2'
    my_authors = my_authors.split('<THREAD_SEP>')
    print(my_authors)
    my_authors = '<ARXIV_SEP>'.join(my_authors)
    print(my_authors)
    my_authors = my_authors.split('<ARXIV_SEP>')
    print(my_authors)
