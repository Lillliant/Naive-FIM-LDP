from itertools import combinations, chain
import math

def calculate_true_frequency(frequency: int, N: int, K: int, P: int, k: int) -> int:
    prob_false_positive = (1-P)/K
    true_frequency = (frequency - math.pow(prob_false_positive, k) * N) / (math.pow(P, k) - math.pow(prob_false_positive, k))
    return true_frequency

# For generating C_k if a list of frequent itemsets are given
# Given elements of length k, generate candidates of length k+1
def generate_candidates_set(Lk: list[tuple[tuple[int], int]]) -> list[tuple[int]]:
    candidates = set()
    Ck = [c[0] for c in Lk] # get the frequent itemsets
    for i, c1 in enumerate(Ck[:-1]):
        for c2 in Ck[i+1:]:
            candidates.add(tuple(set(c1).union(set(c2)))) # union of two frequent k-itemsets
    return prune_candidates_set(list(candidates), Lk)

# Based on the Apriori property, if any subset of a candidate is not frequent, then the candidate cannot be frequent
def prune_candidates_set(candidates: list[tuple[int]], Lk: list[tuple[int]]) -> list[tuple[int]]:
    Lk = [l[0] for l in Lk] # get the frequent itemsets without the support
    for candidate in candidates:
        for i in range(len(candidate)):
            subset = tuple(candidate[:i] + candidate[i+1:]) # generate all subsets of the candidate
            if subset not in Lk: # if any subset of the candidate is not frequent
                candidates.remove(candidate) # remove the candidate
                break
    return candidates

# Generate L_k+1 based on C_k
def next_frequent_itemset(data: list[tuple[int]], C_k: list[tuple[int]], min_support: int, N: int, K: int, P: int, k: int) -> list[tuple[tuple[int], int]]:
    support_set = count_support(data, C_k) # count the support of each candidate within the transactions
    #print("support_set: ", support_set)
    Lk_1 = [(l, s) for (l, s) in support_set.items() if calculate_true_frequency(s, N, K, P, k) >= min_support]
    return Lk_1

# Count of support of each candidate within the transactions
def count_support(data: list[tuple[int]], candidates_set: list[tuple[int]]) -> dict[tuple[int], int]:
    support_set = {candidate: 0 for candidate in candidates_set}
    for transaction in data:
        for candidate in candidates_set:
            if set(candidate).issubset(set(transaction)): # if the candidate is in the transaction
                support_set[candidate] += 1
    return support_set

# Main Apriori algorithm
def LDPapriori(data: list[list[int]], min_support: int, N: int, K: int, P: int) -> list[tuple[tuple[int], int]]:
    frequent_itemsets = []
    k = 1
    C_k = [tuple(e) for e in combinations(set(chain(*data)), 1)] # Generate C1 = all items
    #print("C1: ", C_k)
    L_k = next_frequent_itemset(data, C_k, min_support, N, K, P, k) # Generate L1 = frequent items
    frequent_itemsets.extend(L_k) # Add the frequent items to the itemsets
    #print("L1: ", L_k)
    while L_k: # while Lk is not empty
        # generate candidate sets from Lk
        print("Iteration: ", k)
        C_k = generate_candidates_set(L_k) # Generate Ck+1
        print(f"C{k+1} generated.")
        L_k = next_frequent_itemset(data, C_k, min_support, N, K, P, k) # Generate Lk+1
        print(f"L{k+1} generated.")
        frequent_itemsets.extend(L_k) # Add the frequent items to the itemsets
        k += 1
    
    return frequent_itemsets

if __name__ == "__main__": # test code
    import pprint
    data = [[3, 4, 1], [2, 4, 3], [2, 1], [2, 3], [2, 1, 3, 4]]
    min_support = 2
    frequent_itemset = LDPapriori(data, min_support, 5, 4, 0.5)
    pprint.pprint(frequent_itemset)