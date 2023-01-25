import sys
import itertools

import numpy as np
import torch
from torch_geometric.data import HeteroData
from data_preprocessing import *


# user (twitter); venue
def create_nodes_and_edges(df, include_authors=False):
    node_id_dict = {'paper': [],
                    'tweet': [],
                    'user': []}
    node_dict = {'paper': [],
                 'tweet': [],
                 'user': [[], [], [], [], [], [], []]}
    edge_dict = {('tweet', 'cites', 'paper'): [[], []],
                 ('tweet', 'replies_to', 'tweet'): [[], []],
                 ('user', 'posts', 'tweet'): [[], []],
                 ('tweet', 'mentions', 'user'): [[], []],
                 ('paper', 'shares_author_with', 'paper'): [[], []]}
    label_dict_likes = {'tweet': []}
    label_dict_retweets = {'tweet': []}
    if include_authors:
        node_id_dict['author'] = []
        node_dict['author'] = []
        edge_dict[('author', 'wrote', 'paper')] = [[], []]

    is_target_node = []
    author_paper_dict = {}

    for index, row in df.iterrows():
        # print(row)
        tweets_text = split_row_text(row['text'])
        tweets_like_count = numeric_list_string_to_list(row['like_count'])
        tweets_reply_count = numeric_list_string_to_list(row['reply_count'])
        tweets_retweet_count = numeric_list_string_to_list(row['retweet_count'])

        paper_titles = sep_arxive_data(str(row['arxiv.title']))
        paper_summaries = sep_arxive_data(str(row['arxiv.summary']))

        user_verified = 1 if row['author.verified'] else 0
        user_followers = row['author.followers_count']
        user_following = row['author.following_count']
        user_tweet_count = row['author.tweet_count']
        user_favourites_count = row['author.favourites_count']
        user_listed_count = row['author.listed_count']
        user_description = row['author.description']

        authors = sep_arxive_data(str(row['arxiv.authors']))
        paper_ids = list_string_to_list(row['arxiv_identifiers'])
        tweet_ids = split_tweet_ids(row['tweet_id'])
        user_id = row['author.username'].strip()

        # adding tweets
        for tweet_id_idx, tweet_id in enumerate(tweet_ids):
            if tweet_id not in node_id_dict['tweet']:
                node_id_dict['tweet'].append(tweet_id)
                if tweet_id_idx == 0:
                    is_target_node.append(True)
                else:
                    is_target_node.append(False)
                # tweet features
                node_dict['tweet'].append(tweets_text[tweet_id_idx])
                # node_dict['tweet'].append(np.array([1, 1, 1]))
                label_dict_likes['tweet'].append(tweets_like_count[tweet_id_idx])
                label_dict_retweets['tweet'].append(tweets_retweet_count[tweet_id_idx])
            else:
                if tweet_id_idx == 0:
                    idx_tweet_node = node_id_dict['tweet'].index(tweet_id)
                    is_target_node[idx_tweet_node] = True

        '''
        # adding authors
        if include_authors:
            for authors_thread_id, authors_thread in enumerate(author_cell):
                if '<ARXIV_SEP>' in authors_thread:
                    comment_authors = authors_thread.split('<ARXIV_SEP>')
                    for authors in comment_authors:
                        for author in authors.split('\t'):
                            if author not in node_id_dict['author']:
                                node_id_dict['author'].append(author.strip())
                                # author features
                                node_dict['author'].append(1)
                else:
                    for author in authors_thread.split('\t'):
                        if author not in node_id_dict['author']:
                            node_id_dict['author'].append(author.strip())
                            # author features
                            node_dict['author'].append(1)
        '''
        paper_ids_flattened = list(itertools.chain.from_iterable(paper_ids))
        for idx, val in enumerate(paper_ids_flattened):
            try:
                set_of_authors = authors[idx].split('\t')
            except:
                continue
            for single_author in set_of_authors:
                if single_author in list(author_paper_dict.keys()):
                    author_paper_dict[single_author].append(val)
                else:
                    author_paper_dict[single_author] = [val]

        # adding papers
        papers_added = 0
        for paper_id_idx, paper_id_list in enumerate(paper_ids):
            for paper_id in paper_id_list:
                if paper_id not in node_id_dict['paper']:
                    node_id_dict['paper'].append(paper_id)
                    # paper features
                    node_dict['paper'].append(paper_titles[papers_added] + paper_summaries[papers_added])
                    # node_dict['paper'].append(np.array([0, 0, 0]))
                papers_added += 1

        # adding users
        if user_id not in node_id_dict['user']:
            node_id_dict['user'].append(user_id)
            # user features
            node_dict['user'][0].append(user_verified)
            node_dict['user'][1].append(user_following)
            node_dict['user'][2].append(user_followers)
            node_dict['user'][3].append(user_tweet_count)
            node_dict['user'][4].append(user_favourites_count)
            node_dict['user'][5].append(user_listed_count)
            node_dict['user'][6].append(user_description)
            # np.array([user_verified, user_following, user_followers, user_tweet_count,
            # user_favourites_count, user_listed_count])) # user description

    author_paper_dict = {key: list(set(val)) for key, val in author_paper_dict.items() if len(list(set(val))) > 1}

    for index, row in df.iterrows():
        # print(row)
        author_cell = str(row['arxiv.authors']).split('<THREAD_SEP>')
        paper_ids = list_string_to_list(row['arxiv_identifiers'])
        tweet_ids = split_tweet_ids(row['tweet_id'])
        referenced_tweets_ids = list_string_to_list(row['referenced_tweets_ids'])
        user_id = row['author.username'].strip()
        user_mentions = list_string_to_list(row['mentions'])

        # 'user', 'posts', 'tweet'
        user_node_idx = node_id_dict['user'].index(user_id)
        original_tweet_node_idx = node_id_dict['tweet'].index(tweet_ids[0])

        edge_dict[('user', 'posts', 'tweet')][0].append(user_node_idx)
        edge_dict[('user', 'posts', 'tweet')][1].append(original_tweet_node_idx)

        '''
        # 'author', 'wrote', 'paper'
        if include_authors:
            author_cell = '<ARXIV_SEP>'.join(author_cell)
            author_lists = author_cell.split('<ARXIV_SEP>')
            num_referenced_paper = 0
            for paper_id_list_idx, paper_id_list in enumerate(paper_ids):
                for paper_id in paper_id_list:
                    set_of_authors = author_lists[num_referenced_paper].split('\t')
                    num_referenced_paper += 1
                    node_idx_paper = node_id_dict['paper'].index(paper_id)
                    for author in set_of_authors:
                        node_idx_author = node_id_dict['author'].index(author.strip())
                        edge_dict[('author', 'wrote', 'paper')][0].append(node_idx_author)
                        edge_dict[('author', 'wrote', 'paper')][1].append(node_idx_paper)
        '''

        # 'tweet', 'cites', 'paper'
        # 'tweet', 'mentions', 'user'
        for tweet_index, tweet_id in enumerate(tweet_ids):
            node_idx_tweet = node_id_dict['tweet'].index(tweet_id)
            paper_list_for_tweet = paper_ids[tweet_index]
            if len(paper_list_for_tweet) > 0:
                for paper_id in paper_list_for_tweet:
                    node_idx_paper = node_id_dict['paper'].index(paper_id)
                    edge_dict[('tweet', 'cites', 'paper')][0].append(node_idx_tweet)
                    edge_dict[('tweet', 'cites', 'paper')][1].append(node_idx_paper)
            mentioned_users = user_mentions[tweet_index]
            if len(mentioned_users) > 0:
                for mentioned_user in mentioned_users:
                    mentioned_user = mentioned_user.strip()
                    if mentioned_user in node_id_dict['user']:
                        node_idx_user = node_id_dict['user'].index(mentioned_user)
                        edge_dict[('tweet', 'mentions', 'user')][0].append(node_idx_tweet)
                        edge_dict[('tweet', 'mentions', 'user')][1].append(node_idx_user)

        # 'tweet', 'replies_to', 'tweet'
        if len(tweet_ids) > 1:
            for tweet_idx, tweet_id in enumerate(tweet_ids):
                if tweet_idx == 0:
                    continue
                else:
                    node_idx_tweet_from = node_id_dict['tweet'].index(tweet_id)
                    tweet_ids_to = referenced_tweets_ids[tweet_idx]
                    for tweet_id_to in tweet_ids_to:
                        if tweet_id_to in node_id_dict['tweet']:
                            node_idx_tweet_to = node_id_dict['tweet'].index(tweet_id_to)
                            edge_dict[('tweet', 'replies_to', 'tweet')][0].append(node_idx_tweet_from)
                            edge_dict[('tweet', 'replies_to', 'tweet')][1].append(node_idx_tweet_to)

        # 'paper', 'shares_author_with', 'paper'
        for author, paper_list in author_paper_dict.items():
            for i in range(len(paper_list)):
                node_idx_paper_one = node_id_dict['paper'].index(paper_list[i])
                for k in range(i, len(paper_list)):
                    if i != k:
                        node_idx_paper_two = node_id_dict['paper'].index(paper_list[k])
                        edge_dict[('paper', 'shares_author_with', 'paper')][0].append(node_idx_paper_one)
                        edge_dict[('paper', 'shares_author_with', 'paper')][1].append(node_idx_paper_two)

    return node_id_dict, node_dict, edge_dict, label_dict_likes, label_dict_retweets, is_target_node


def save_graph(graph, file_name='graph.pt'):
    torch.save(graph, file_name)


def create_heterogeneous_graph(x_dict, edge_index_dict, label_dict_likes=None, label_dict_retweets=None,
                               is_target_node=None):
    x_dict = {node_type: torch.FloatTensor(np.array(x)) for node_type, x in x_dict.items()}
    edge_index_dict = {edge_type: torch.LongTensor(np.array(edges)) for edge_type, edges in edge_index_dict.items()}
    label_dict_likes = {node_type: torch.Tensor(label) for node_type, label in label_dict_likes.items()}
    label_dict_retweets = {node_type: torch.Tensor(label) for node_type, label in label_dict_retweets.items()}
    graph = HeteroData()

    for key, val in x_dict.items():
        graph[key].x = val
    for key, val in edge_index_dict.items():
        graph[key].edge_index = val
    for key, val in label_dict_likes.items():
        graph[key]['likes'] = val
    for key, val in label_dict_retweets.items():
        graph[key]['retweets'] = val
    graph['tweet']['target_nodes'] = torch.Tensor(is_target_node)

    return graph


if __name__ == '__main__':
    df = read_data('valid')
    print(df.shape)
    print(10 * '*')
    node_id_dict, node_dict, edge_dict, label_dict_likes, label_dict_retweets, is_target_node = create_nodes_and_edges(df, include_authors=False)
    # print(len(node_id_dict['author'])) #, np.unique(np.array(node_id_dict['author'])).shape)
    print(len(node_id_dict['tweet']), len(node_dict['tweet'])) #, np.unique(np.array(node_id_dict['tweet'])).shape)
    print(len(node_id_dict['paper']), len(node_dict['paper'])) #, np.unique(np.array(node_id_dict['paper'])).shape)
    print(len(node_id_dict['user'][0]), len(node_dict['user'][0])) #, np.unique(np.array(node_id_dict['user'])).shape)
    print(10*'*')
    # print(len(edge_dict[('author', 'wrote', 'paper')][0]))
    print(len(edge_dict[('tweet', 'cites', 'paper')][0]))
    print(len(edge_dict[('tweet', 'replies_to', 'tweet')][0]))
    print(len(edge_dict[('user', 'posts', 'tweet')][0]))
    print(len(edge_dict[('tweet', 'mentions', 'user')][0]))
    print(len(edge_dict[('paper', 'shares_author_with', 'paper')][0]))
    print(10 * '*')
    print(len(label_dict_likes['tweet']))
    print(len(label_dict_retweets['tweet']))
    print(len(is_target_node), 'True:', sum(is_target_node))

    # concat and transform user features
    for idx, feature_list in enumerate(node_dict['user']):
        if idx < len(node_dict['user'])-1:
            scaled_features = np.array(feature_list) / (max(feature_list) if max(feature_list) > 0 else 1)
            node_dict['user'][idx] = scaled_features[:, np.newaxis]
        else:
            node_dict['user'][idx] = text_features_to_numeric(feature_list)
    node_dict['user'] = np.concatenate(node_dict['user'], axis=1)

    # convert text features to numeric
    node_dict['paper'] = text_features_to_numeric(node_dict['paper'], text_type='paper')
    node_dict['tweet'] = text_features_to_numeric(node_dict['tweet'])

    graph = create_heterogeneous_graph(x_dict=node_dict,
                                       edge_index_dict=edge_dict,
                                       label_dict_likes=label_dict_likes,
                                       label_dict_retweets=label_dict_retweets,
                                       is_target_node=is_target_node)
    print(graph)
    # save_graph(graph)
