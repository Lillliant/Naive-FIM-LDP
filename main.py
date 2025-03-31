import datetime
from algorithms.apriori import apriori
from algorithms.eclat import eclat
from algorithms.LDPapriori import LDPapriori
import os
import random, pickle
import math
from itertools import chain

# Global variables
MIN_SUPPORT = 900
SIZE = 9976 # N: Number of transactions in the dataset
PRIVACY_BUDGET = 7
PERTURB = True
P = 0 # Probability of returning true item
K = 0 # Number of items in the dataset

def perturb_function(data: list[list[int]]) -> list[list[int]]:
    random.seed(0)
    euler_exp = math.exp(PRIVACY_BUDGET)
    items = [e for e in set(chain(*data))]
    K = len(items) # number of items in the dataset
    P =  euler_exp / (euler_exp + K - 1) # probability of returning true item
    print(f"Probability of returning true value: {P}")
    for transaction in data:
        for item in transaction:
            if (random.random() > P): # 1-p probability of returning false item
                idx = items.index(item) # get the index of the item
                other_items = items[:idx]
                if idx + 1 < N: other_items.extend(items[idx + 1:]) # get the other items

                selected_item = random.choice(other_items) # Randomly select an item from the dataset
                # Check if the selected item is already in the transaction
                while selected_item in transaction:
                    selected_item = random.choice(other_items)

                transaction[transaction.index(item)] = selected_item # replace the item with the selected item
    print("Perturbation done.")
    return data

def evaluate():
    pass

def load_data(input_path: str, output_dir: str, size: int = None, perturb: bool = False) -> list[list[int]]:
    random.seed(0)
    file_name = 'perturbed_data.pkl' if perturb else 'data.pkl'
    if os.path.exists(f'{output_dir}/{file_name}'):
        print("Data already exists. Loading data...")
        data = pickle.load(open(f'{output_dir}/{file_name}', 'rb'))
        data = random.sample(data, size) if size is not None else data
    else:
        print("Data not found. Loading from file...")
        data = []
        with open(input_path, 'r') as f:
            while line := f.readline():
                data.append(list(map(int, line.strip().split())))
        data = random.sample(data, size) if size is not None else data
        data = perturb_function(data) if perturb else data
        pickle.dump(data, open(f'{output_dir}/{file_name}', 'wb'))
    
    print(f"Data loaded. Number of transactions: {len(data)}")
    return data

def write_output(output_path: str, algorithm: str, frequent_itemset: list[tuple[int]], start_time: datetime.datetime, end_time: datetime.datetime):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_path + f'{algorithm}.txt', 'w') as f:
        for itemset in frequent_itemset:
            f.write(str(itemset) + '\n')
        f.write("Total number of frequent itemsets: " + str(len(frequent_itemset)) + '\n')
        f.write("Start time: " + str(start_time) + '\n')
        f.write("End time: " + str(end_time) + '\n')
        f.write("Total time: " + str(end_time - start_time) + '\n')

def frequent_mining(output_path: str, data: list[list[int]], algorithm: str, min_support: int) -> list[tuple[int]]:
    # Processing for Apriori
    print(f"Processing with {algorithm.title()}...")
    stime = datetime.datetime.now()
    print("Start time: ", stime)
    match algorithm:
        case 'apriori':
            frequent_itemset = apriori(data, min_support)
        case 'eclat':
            frequent_itemset = eclat(data, min_support)
        case 'ldp_apriori':
            frequent_itemset = LDPapriori(data, min_support, SIZE, K, P)
        case 'ldp_eclat':
            frequent_itemset = LDPeclat(data, min_support, SIZE, K, P)
        case _:
            raise ValueError("Invalid algorithm specified.")
    etime = datetime.datetime.now()
    print("End time: ", etime)
    write_output(output_path, algorithm, frequent_itemset, stime, etime)
    return frequent_itemset

if __name__ == '__main__':
    # Set the output path and minimum support
    output_path = f'./output/t_{SIZE}_s{MIN_SUPPORT}{'_perturb' if PERTURB else ''}/'
    data_path = './data/t25i10d10k/t25i10d10k.txt'
    data = load_data(data_path, output_path, SIZE, perturb=PERTURB)
    
    #apriori_fim = frequent_mining(output_path, data, 'apriori', MIN_SUPPORT)
    #eclat_fim = frequent_mining(output_path, data, 'eclat', MIN_SUPPORT)
