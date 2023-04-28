import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import random
with open('database/store_data.csv') as f:
    records = []
    for line in f:
        records.append(line.strip().split(','))

desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 10)

#length = []
# for i in records:
#     length.append(len(i))
# print(sum(length)/len(length))

te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)
# print(df.describe())

# Selecting 4% as min support as it is the % of the average
# length of transaction (4) against number of distinct items rounded up
#frequent_itemsets = apriori(df, min_support=0.04, use_colnames=True)
#frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
#print(frequent_itemsets[(frequent_itemsets['length'] == 2)])
#print(association_rules(frequent_itemsets, metric='confidence', min_threshold=0.04))
list1 = []
list2 = []
count = 0
for i in records:
    count = random.randint(0,1)
    if count == 0:
        list1.append(i)
        count = 1
    else:
        list2.append(i)
        count = 0
print(len(list1),len(list2))

te_ary = te.fit(list1).transform(list1)
df1 = pd.DataFrame(te_ary, columns=te.columns_)

te_ary = te.fit(list2).transform(list2)
df2 = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets1 = apriori(df1, min_support=0.04, use_colnames=True)
frequent_itemsets1['length'] = frequent_itemsets1['itemsets'].apply(lambda x: len(x))
print(association_rules(frequent_itemsets1, metric='confidence', min_threshold=0.04))#print(association_rules(frequent_itemsets, metric='confidence', min_threshold=0.04))

frequent_itemsets2 = apriori(df2, min_support=0.04, use_colnames=True)
frequent_itemsets2['length'] = frequent_itemsets2['itemsets'].apply(lambda x: len(x))
print('\n\n\\n\n\n\n\n\n')
print(association_rules(frequent_itemsets2, metric='confidence', min_threshold=0.04))#print(association_rules(frequent_itemsets, metric='confidence', min_threshold=0.04))

