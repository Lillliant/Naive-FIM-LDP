import datetime
from algorithms.LDPeclat import LDPeclat
from algorithms.apriori import apriori
from algorithms.eclat import eclat
from algorithms.LDPapriori import LDPapriori
import os
import random, pickle
import math
from itertools import chain

# Global variables
MIN_SUPPORT = 20
SIZE = None # N: Number of transactions in the dataset
PRIVACY_BUDGET = 7

def perturb_function(data: list[list[int]]) -> list[list[int]]:
    random.seed(0)
    euler_exp = math.exp(PRIVACY_BUDGET)
    items = [e for e in set(chain(*data))]
    global P, K
    K = len(items) # number of items in the dataset
    P =  euler_exp / (euler_exp + K - 1) # probability of returning true item
    new_data = [] # create a copy as data is passed by reference
    for transaction in data:
        new_transaction = []
        for item in transaction:
            if (random.random() > P): # 1-p probability of returning false item
                idx = items.index(item) # get the index of the item
                other_items = items[:idx]
                if idx + 1 < K: other_items.extend(items[idx + 1:]) # get the other items
                selected_item = random.choice(other_items) # Randomly select an item from the dataset
                # Check if the selected item is already in the transaction
                while selected_item in transaction:
                    selected_item = random.choice(other_items)
                new_transaction.append(selected_item) # replace the item with the selected item
            else:
                new_transaction.append(item)
        new_data.append(new_transaction)
    return new_data

def evaluate(y_true: list[tuple[int]], y_pred: list[tuple[int]], algorithm1: str, algorithm2: str):
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score
    
    print(f"Evaluating {algorithm1} and {algorithm2}...")
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

def load_data(input_path: str, output_dir: str, size: int = None, perturb: bool = False) -> list[list[int]]:
    random.seed(0)
    file_name = 'perturbed_data.pkl' if perturb else 'data.pkl'
    if os.path.exists(f'{output_dir}/{file_name}'):
        print(f'Data already exists. Loading {file_name}...')
        data = pickle.load(open(f'{output_dir}/{file_name}', 'rb'))
        if perturb:
            stat = pickle.load(open(f'{output_dir}/perturbed_stat.pkl', 'rb'))
            global P, K
            P = stat[0]
            K = stat[1]
        return data
    else:
        if not os.path.exists(output_dir): os.makedirs(output_dir) # in case the output directory does not exist
        print("Data not found. Loading from file...")
        data = []
        with open(input_path, 'r') as f:
            while line := f.readline():
                data.append(list(map(int, line.strip().split())))
        original_data = random.sample(data, size) if size is not None else data # Sample the data, if size is specified
        # Save the data and perturbed data version for consistency
        pickle.dump(original_data, open(f'{output_dir}/data.pkl', 'wb'))
        perturbed_data = perturb_function(data)
        pickle.dump(perturbed_data, open(f'{output_dir}/perturbed_data.pkl', 'wb'))
        with open(f'{output_dir}/perturbed_stat.txt', 'w') as f: # save the statistics of the perturbed data in readable format
            f.write("Privacy budget: " + str(PRIVACY_BUDGET) + '\n')
            f.write("Number of transactions (N): " + str(SIZE) + '\n')
            f.write("Number of items (K): " + str(K) + '\n')
            f.write("Probability (P): " + str(P) + '\n')
            f.close()
        pickle.dump([P, K], open(f'{output_dir}/perturbed_stat.pkl', 'wb')) # save the statistics of the perturbed data that can be used for future algorithm runs
        return perturbed_data if perturb else original_data

def write_output(output_path: str, algorithm: str, frequent_itemset: list[tuple[int]], start_time: datetime.datetime, end_time: datetime.datetime):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Make readable output
    with open(output_path + f'/{algorithm}.txt', 'w') as f:
        for itemset in frequent_itemset:
            print(itemset)
            f.write(str(itemset) + '\n')
        f.write("Total number of frequent itemsets: " + str(len(frequent_itemset)) + '\n')
        f.write("Minimum support: " + str(MIN_SUPPORT) + '\n')
        f.write("Start time: " + str(start_time) + '\n')
        f.write("End time: " + str(end_time) + '\n')
        f.write("Total time: " + str(end_time - start_time) + '\n')
        
    # Make output for evaluation
    pickle.dump(frequent_itemset, open(output_path + f'/{algorithm}_fim.pkl', 'wb'))
    pickle.dump(end_time - start_time, open(output_path + f'/{algorithm}_time.pkl', 'wb'))

def frequent_mining(output_path: str, data_path: str, algorithm: str, min_support: int) -> list[tuple[int]]:
    # Processing for Apriori
    print(f"Processing with {algorithm.title()}...")
    match algorithm:
        case 'apriori':
            data = load_data(data_path, output_path, SIZE, perturb=False)
            stime = datetime.datetime.now()
            print("Start time: ", stime)
            frequent_itemset = apriori(data, min_support)
        case 'eclat':
            data = load_data(data_path, output_path, SIZE, perturb=False)
            stime = datetime.datetime.now()
            print("Start time: ", stime)
            frequent_itemset = eclat(data, min_support)
        case 'ldp_apriori':
            data = load_data(data_path, output_path, SIZE, perturb=True)
            stime = datetime.datetime.now()
            print("Start time: ", stime)
            print("Number of items (K): ", K)
            print("Probability (P): ", P)
            frequent_itemset = LDPapriori(data, min_support, SIZE, K, P)
        case 'ldp_eclat':
            data = load_data(data_path, output_path, SIZE, perturb=True)
            stime = datetime.datetime.now()
            print("Start time: ", stime)
            print("Number of items (K): ", K)
            print("Probability (P): ", P)
            frequent_itemset = LDPeclat(data, min_support, SIZE, K, P)
        case _:
            raise ValueError("Invalid algorithm specified.")
    etime = datetime.datetime.now()
    print("End time: ", etime)
    write_output(output_path, algorithm, frequent_itemset, stime, etime)
    return frequent_itemset

if __name__ == '__main__':
    # Set the output path and minimum support
    output_path = f'./output/t_{SIZE}_s{MIN_SUPPORT}_p{PRIVACY_BUDGET}'
    data_path = './data/toy/toy1.txt' #./data/t25i10d10k/t25i10d10k.txt
    apriori_fim = frequent_mining(output_path, data_path, 'apriori', MIN_SUPPORT)
    eclat_fim = frequent_mining(output_path, data_path, 'eclat', MIN_SUPPORT)
    #ldpeclat_fim = frequent_mining(output_path, data_path, 'ldp_eclat', MIN_SUPPORT)
    #ldpapriori_fim = frequent_mining(output_path, data_path, 'ldp_apriori', MIN_SUPPORT)
    
    # Evaluate the results
    #evaluate(apriori_fim, ldpapriori_fim, 'apriori', 'ldp_apriori')
    #evaluate(eclat_fim, ldpeclat_fim, 'eclat', 'ldp_eclat')
