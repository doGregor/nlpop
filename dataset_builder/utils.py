import sys

from tqdm import tqdm
import math
import arxiv
import time

def fetch_arxiv_data(arxiv_identifiers_list, k=100):
    # using up to k inputs due to unexpected error in arxiv library
    response = list()
    for i in tqdm(range(math.ceil(len(arxiv_identifiers_list)/k))):
        search = arxiv.Search(
        id_list = arxiv_identifiers_list[k*i:(i+1)*k]
        )
        response.extend(list(search.results()))
        time.sleep(1)
    return response

column_names = ['entry_id', 'updated', 'published', 'title', 'authors', 'summary', 'comment',
                'journal_ref', 'doi', 'primary_category', 'categories']

def parse_result(result, column_names=column_names):
    entry_id = result.entry_id
    updated = result.updated
    published = result.published
    title = result.title
    authors = [subresult.name for subresult in result.authors]
    try:
        authors = '\t'.join(authors)
    except:
        pass
    summary = result.summary
    comment = result.comment
    journal_ref = result.journal_ref
    doi = result.doi
    primary_category = result.primary_category
    categories = result.categories
    try:
        categories = '\t'.join(categories)
    except:
        pass

    rows = [entry_id, updated, published, title, authors, summary, comment, journal_ref, doi,
           primary_category, categories]
    return rows
    '''
    content = {'id': entry_id, 'updated': updated, 'published': published, 'title': title, 'authors': authors,
               'summary': summary, 'comment': comment, 'journal_ref': journal_ref, 'doi': doi,
               'category': primary_category, 'categories': categories}
    return content
    '''

def make_dict(column_names, rows):
    pd_dict = dict()
    for i, column_name in enumerate(column_names):
        pd_dict[column_name] = rows[:,i]
    
    return pd_dict

