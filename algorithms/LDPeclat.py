
import math

def calculate_true_frequency(frequency: int, N: int, K: int, P: float, k: int) -> int:
    prob_false_positive = (1-P)/K
    #print(f'P: {P}')
    #print(f"Numerator: {frequency - math.pow(prob_false_positive, k) * N}")
    #print(f"Denominator: {math.pow(P, k) - math.pow(prob_false_positive, k)}")
    true_frequency = (frequency - math.pow(prob_false_positive, k) * N) / abs((math.pow(P, k) - math.pow(prob_false_positive, k)))
    #print(f"True frequency: {true_frequency}")
    return true_frequency

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

def next_depth_candidates(init_element: tuple[tuple[int], set[int]], combinations: list[tuple[int], set[int]], min_support: int, N: int, K: int, P: float, k: int) -> list[tuple[int]]:
    frequent_itemsets = []
    for i, tid in combinations:
        candidate = tuple(set(init_element[0]).union(set(i)))
        c_tid = init_element[1].intersection(tid)
        if calculate_true_frequency(len(c_tid), N, K, P, k) >= min_support:
            frequent_itemsets.append((candidate, c_tid))
    return frequent_itemsets

def next_frequent_itemset(C_k: list[tuple[int]], min_support: int, N: int, K: int, P: float, k: int) -> list[tuple[int]]:
    candidate_depth = []
    for i, c in enumerate(C_k[:-1]):
        candidate_depth.extend(next_depth_candidates(c, C_k[i+1:], min_support, N, K, P, k))
    # remove duplicates
    frequent_depth = []
    for item in candidate_depth:
        if item not in frequent_depth:
            frequent_depth.append(item)
    return frequent_depth

# Main Eclat algorithm
def LDPeclat(data: list[list[int]], min_support: int, N: int, K: int, P: float) -> list[tuple[tuple[int], int]]:
    frequent_itemsets = []
    C_1 = generate_item_tid(data) # generate the item-transaction table
    #print("C_1: ", C_1)
    L_1 = [(c, t) for c, t in C_1.items() if calculate_true_frequency(len(t), N, K, P, 1) >= min_support] # generate the frequent items
    frequent_itemsets.extend([(k, len(v)) for k, v in L_1]) # add the frequent items to the itemsets

    for i, (c, tid) in enumerate(L_1[:-1]):
        print("Iterating for: ", c)
        C_k = next_depth_candidates((c, tid), L_1[i+1:], min_support, N, K, P, 2)
        frequent_itemsets.extend([(k, len(v)) for k, v in C_k])
        k = 3
        while C_k: # while Ck is not empty
            #print("Iterating for: ", C_k)
            # calculate for each depth 'path'd
            L_k = next_frequent_itemset(C_k, min_support, N, K, P, k) # generate Lk
            frequent_itemsets.extend([(k, len(v)) for k, v in L_k])
            C_k = L_k
    return frequent_itemsets

if __name__ == "__main__":
    import pprint
    data = [[3, 4, 1], [2, 4, 3], [2, 1], [2, 3], [2, 1, 3, 4]]
    min_support = 2
    frequent_itemset = LDPeclat(data, min_support, 5, 4, 0.5)
    pprint.pprint(frequent_itemset)