# convert the transaction data into vertical format
# (i.e., each item is associated with the set of transactions in which it appears)
def generate_item_tid(data: list[list[int]]) -> dict[tuple[int], set[int]]:
    candidate_transaction_set = {}
    for idx, transaction in enumerate(data):
        for item in transaction:
            if candidate_transaction_set.get((item,)) is None:
                candidate_transaction_set[(item,)] = set()
                candidate_transaction_set[(item,)].add(idx)
            else:
                candidate_transaction_set[(item,)].add(idx)
    return candidate_transaction_set

def next_breadth_candidates(init_element: tuple[tuple[int], set[int]], combinations: list[tuple[int], set[int]], min_support: int) -> list[tuple[int]]:
    frequent_itemsets = []
    for i, tid in combinations:
        candidate = tuple(set(init_element[0]).union(set(i)))
        c_tid = init_element[1].intersection(tid)
        if len(c_tid) >= min_support:
            frequent_itemsets.append((candidate, c_tid))
    return frequent_itemsets

def next_frequent_itemset(C_k: list[tuple[int]], min_support: int) -> list[tuple[int]]:
    frequent_breadth = []
    for i, c in enumerate(C_k[:-1]):
        frequent_breadth.extend(next_breadth_candidates(c, C_k[i+1:], min_support))
    return frequent_breadth

# Main Eclat algorithm
def eclat(data: list[list[int]], min_support: int) -> list[tuple[tuple[int], int]]:
    frequent_itemsets = []
    C_1 = generate_item_tid(data) # generate the item-transaction table
    L_1 = [(k, v) for k, v in C_1.items() if len(v) >= min_support] # generate the frequent items
    frequent_itemsets.extend([(k, len(v)) for k, v in L_1]) # add the frequent items to the itemsets

    for i, (c, tid) in enumerate(L_1[:-1]):
        print("Iterating for: ", c)
        C_k = next_breadth_candidates((c, tid), L_1[i+1:], min_support)
        frequent_itemsets.extend([(k, len(v)) for k, v in C_k])
        while C_k: # while Ck is not empty
            # calculate for each breadth
            L_k = next_frequent_itemset(C_k, min_support) # generate Lk
            frequent_itemsets.extend([(k, len(v)) for k, v in L_k])
            C_k = L_k
    return frequent_itemsets

if __name__ == "__main__":
    import pprint
    data = [[3, 4, 1], [2, 4, 3], [2, 1], [2, 3], [2, 1, 3, 4]]
    min_support = 2
    frequent_itemset = eclat(data, min_support)
    pprint.pprint(frequent_itemset)