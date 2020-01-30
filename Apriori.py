# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import fisher_exact
import warnings
warnings.filterwarnings('ignore')

def get_transaction_data():
    """Load groceries transaction data into DataFrame"""
    df = pd.read_csv('Sample dataset.csv', encoding='latin-1')
    df = df[['SOC_NAME', 'PREVAILING_WAGE', 'H-1B_DEPENDENT', 'WORKSITE_STATE', 'CASE_STATUS']].copy()
    df['CASE_STATUS'] = df['CASE_STATUS'].replace(['WITHDRAWN'], ['DENIED'])
    df['CASE_STATUS'] = df['CASE_STATUS'].replace(['CERTIFIEDWITHDRAWN'], ['CERTIFIED'])
    df = df.stack().reset_index(-1, drop=True)
    df.index.names = ['id']
    df = pd.get_dummies(df, prefix='', prefix_sep='')
    return df.groupby(level='id').sum()

def prune_candidates(all_txn, candidates, candidate_size, min_support):
    """Return DataFrame with itemsets of candidate_size with min_support
        all_txn: numpy array of transaction-product matrix
        candidates: list of tuples containing product id
        candidate_size: length of item set
        min_support: support threshold
        """
    itemsets = {}
    for candidate in candidates:
        candidate_txn = all_txn[:, candidate].reshape(-1, candidate_size)  
        relevant_txn = candidate_txn[(candidate_txn == 1).all(axis=1)]
        candidate_support = relevant_txn.shape[0] / all_txn.shape[0]
        if candidate_support >= min_support:
            itemsets[frozenset(candidate)] = candidate_support
    result = pd.Series(itemsets).to_frame('support')
    return result.assign(length=candidate_size)

def find_association_rules(itemsets, n_txn, n_items, min_confidence=0, min_lift=0, min_pvalue=0):
    """Find rules {antecedent} => {consequent} with min_confidence, min_lift and min_pvalue
    itemsets: DataFrame containing all itemsets and their support
    min_confidence, min_lift, min_pvalue: confidence & lift & pvalue thresholds
    """
    support = itemsets.loc[:, 'support'].to_dict()
    new_rules = []
    for itemset in itemsets.loc[itemsets.length == n_items].index:
        for n_antecedents in range(1, n_items):
            antecedents = [frozenset(a)
                           for a in combinations(itemset, r=n_antecedents)]
            for antecedent in antecedents:
                consequent = itemset.difference(antecedent)
                sAC = support[itemset]
                sA, sC = support[antecedent], support[consequent]
                confidence = sAC / sA
                lift = sAC / (sA * sC)
                contingency_table = n_txn * np.array([[sAC, sA - sAC],
                                        [sC - sAC, 1 - sA - sC + sAC]])
                _, p_value = fisher_exact(contingency_table,
                                          alternative='greater')

                if (confidence >= min_confidence) and (lift >= min_lift) and (p_value >= min_pvalue):
                    new_rule = [itemset, antecedent, consequent,
                                support[itemset], support[antecedent], support[consequent],
                                confidence, lift, p_value]
                    new_rules.append(new_rule)
    return new_rules

def create_rule_label(df, item_id):
    antecedent = df.antecedent.apply(lambda x: '{' + ', '.join([item_id[i] for i in x]) + '}')
    consequent = df.consequent.apply(lambda x: '{' + ', '.join([item_id[i] for i in x]) + '}')
    print('1')
    return antecedent.str.cat(consequent, sep=' => ')


def Apriori():
    data = get_transaction_data()
    
    item_id = pd.Series(dict(enumerate(data.columns)))
    print(item_id)
    transactions = data.values
    print(transactions.shape)
    n_txn = transactions.shape[0]

    min_support = 0.01
    item_length = 1
    candidates = list(zip(item_id.index))
    candidates_tested = 0
    itemsets = pd.DataFrame(columns=['support', 'length'])
    new_rules = []
    rule_data = ['itemset', 'antecedent', 'consequent',
                 'support_rule', 'support_antecedent', 'support_consequent',
                 'confidence', 'lift', 'pvalue']
    rules = pd.DataFrame(columns=rule_data)

    for i in range(1, 5):
        remaining_items = np.unique([item for t in candidates for item in t])
        new_candidates = list(combinations(remaining_items, r=i))
        print('Length {}: {:>10,.0f}'.format(i, len(new_candidates)), new_candidates[:5])
        
    while candidates:
        new_items = prune_candidates(transactions, candidates, item_length, min_support)
        itemsets = itemsets.append(new_items)
        if item_length > 1:
            new_rules = find_association_rules(itemsets, n_txn, item_length)
            rules = pd.concat([rules, pd.DataFrame(new_rules, columns=rules.columns)], ignore_index=True)
                
        candidates_tested += len(candidates)
        print('Itemset Length {}\tCandidates: {:>7,.0f}\tNew Items: {:>7,.0f}'.format(item_length, len(candidates), len(new_items)))
        item_length += 1
        remaining_items = np.unique([item for t in new_items.index for item in t])
        candidates = list(combinations(remaining_items, r=item_length))

    rules = rules.apply(pd.to_numeric, errors='ignore')
    
    # Rules with the index values
    print(rules)
    rules['rule'] = create_rule_label(rules, item_id)
    
    # Association Rules with the values corresponding to the indices
    print(rules[['rule', 'support_rule', 'confidence', 'lift', 'pvalue']].sort_values('lift', ascending=False))
    