# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 10:12:48 2020

@author: ajay
"""

#%% Importing function
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#%% Importing data
raw_data=pd.read_table('groceries.csv',header=None)
raw_data.columns= ['products']
raw_data=raw_data.values.tolist()

def product_split(data):
    w_list=[]
    for i in range(len(data)):
        c=str(data[i]).replace("['",'').replace("']",'').split(',')
        w_list.append(c)
    return w_list

# Modifying raw data in list format
dataset=product_split(raw_data) 

def pairwise_mining(data,MIN_COUNT=10,THRESHOLD=0.5,LENGTH=2):
    """
    This function will take dataset as list format and convert it into
    dictionary with itemsets as key and value as confidence score after filtering 
    with defined minimum count and confidence threshold.\n
    
    Input Parameters
    ----------------
    data,MIN_COUNT= 10, THRESHOLD= 0.5, LENGTH= 2:
        data : Dataset in list format \n
        MIN_COUNT : Minimum count of item, this value will be used to compute minimum support \n
        THRESHOLD : Confidence score threshold \n
        LENGTH : length of itemsets \n
    
    Returns
    -------
    Final_pairs:  \n
        This is a dictionary with keys as itemsets and values as confidence score \n
    
    """     
    encoder = TransactionEncoder()
    encoded_data = encoder.fit(data).transform(data)
    df = pd.DataFrame(encoded_data,columns=encoder.columns_)
    
    # association rule applying using mlxtend ibrary with minimum support as described in problem
    df1 = apriori(df,min_support=MIN_COUNT/len(data),use_colnames=True)
    item_set=association_rules(df1, metric="confidence", min_threshold=THRESHOLD)   
    item_set["antecedent_len"] = item_set["antecedents"].apply(lambda x: len(x))
    item_set=item_set[(item_set['antecedent_len']==LENGTH)]
    item_set.reset_index(inplace=True,drop=True)
    # item_set['antecedents']=[list(i) for i in item_set['antecedents']]
    final_pairs = dict(zip(list(item_set['antecedents']),list(item_set['confidence'])))
    return final_pairs

# Final result as described in problem
final_pairs=pairwise_mining(dataset,MIN_COUNT=10,THRESHOLD=0.5,LENGTH=2)
