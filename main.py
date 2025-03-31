import datetime, os, math, pprint, random, pickle
from algorithms.LDPeclat import LDPeclat
from algorithms.apriori import apriori
from algorithms.eclat import eclat
from algorithms.LDPapriori import LDPapriori
from itertools import chain
from util import *

# Global variables
MIN_SUPPORT = 900
SIZE = 9976 # N: Number of transactions in the dataset
PRIVACY_BUDGET = 9

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

# Metrics to Examine: F1 score, Precision, Recall, Execution time, False Positive Rate, False Negative Rate
def eval(output_dir: str, algorithms: list[str]):
    # Create the director to store the evaluation results
    if not os.path.exists(f'{output_dir}/eval/'): os.makedirs(f'{output_dir}/eval/')

    # Metrics to examine: Average percentage of change within each transaction
    change_percentage = 0
    if os.path.exists(f'{output_dir}/data.pkl') and os.path.exists(f'{output_dir}/perturbed_data.pkl'):
        data = pickle.load(open(f'{output_dir}/data.pkl', 'rb'))
        perturbed_data = pickle.load(open(f'{output_dir}/perturbed_data.pkl', 'rb'))
        change_percentage = average_item_change_per_transaction(data, perturbed_data)
    # Metrics to examine: Execution time (expressed in seconds)
    execution_time = {a: 0 for a in algorithms}
    for a in algorithms:
        if os.path.exists(f'{output_dir}/{a}_time.pkl'):
            execution_time[a] = pickle.load(open(f'{output_dir}/{a}_time.pkl', 'rb')).total_seconds()
        else: print(f"'{output_dir}/{a}_time.pkl' not found. Skipping the algorithm...")

    # Use Apriori and Eclat as the baseline for correct frequent itemset mining
    # Either consider the support or just the frequent itemsets
    precision = {}
    s_precision = {}
    recall = {}
    s_recall = {}
    false_positive = {}
    false_negative = {}
    f1 = {}
    s_f1 = {}
    support_deviation = {}
    if 'apriori' in algorithms and 'ldp_apriori' in algorithms:
        apriori_fim = pickle.load(open(f'{output_dir}/apriori_fim.pkl', 'rb'))
        ldp_apriori_fim = pickle.load(open(f'{output_dir}/ldp_apriori_fim.pkl', 'rb'))
        precision['ldp_apriori'] = calculate_precision(apriori_fim, ldp_apriori_fim, s=False)
        s_precision['ldp_apriori'] = calculate_precision(apriori_fim, ldp_apriori_fim, s=True)
        recall['ldp_apriori'] = calculate_recall(apriori_fim, ldp_apriori_fim, s=False)
        s_recall['ldp_apriori'] = calculate_recall(apriori_fim, ldp_apriori_fim, s=True)
        false_positive['ldp_apriori'] = calculate_fp(apriori_fim, ldp_apriori_fim)
        false_negative['ldp_apriori'] = calculate_fn(apriori_fim, ldp_apriori_fim)
        f1['ldp_apriori'] = calculate_f1(apriori_fim, ldp_apriori_fim, s=False)
        s_f1['ldp_apriori'] = calculate_f1(apriori_fim, ldp_apriori_fim, s=True)
        support_deviation['ldp_apriori'] = average_support_deviation(apriori_fim, ldp_apriori_fim)
    if 'eclat' in algorithms and 'ldp_eclat' in algorithms:
        eclat_fim = pickle.load(open(f'{output_dir}/eclat_fim.pkl', 'rb'))
        ldp_eclat_fim = pickle.load(open(f'{output_dir}/ldp_eclat_fim.pkl', 'rb'))
        precision['ldp_eclat'] = calculate_precision(eclat_fim, ldp_eclat_fim, s=False)
        s_precision['ldp_eclat'] = calculate_precision(eclat_fim, ldp_eclat_fim, s=True)
        recall['ldp_eclat'] = calculate_recall(eclat_fim, ldp_eclat_fim, s=False)
        s_recall['ldp_eclat'] = calculate_recall(eclat_fim, ldp_eclat_fim, s=True)
        false_positive['ldp_eclat'] = calculate_fp(eclat_fim, ldp_eclat_fim)
        false_negative['ldp_eclat'] = calculate_fn(eclat_fim, ldp_eclat_fim)
        f1['ldp_eclat'] = calculate_f1(eclat_fim, ldp_eclat_fim, s=False)
        s_f1['ldp_eclat'] = calculate_f1(eclat_fim, ldp_eclat_fim, s=True)
        support_deviation['ldp_eclat'] = average_support_deviation(eclat_fim, ldp_eclat_fim)

    # Save the performance evaluation metrics
    performance_metrics = {
        'execution_time': execution_time,
        'precision': precision,
        's_precision': s_precision,
        'recall': recall,
        's_recall': s_recall,
        'false_positive_rate': false_positive,
        'false_negative_rate': false_negative,
        'f1': f1,
        's_f1': s_f1,
        'average_support_deviation': support_deviation,
        'average_transaction_change_percentage': change_percentage
    }

    print("Performance metrics:")
    pprint.pprint(performance_metrics)
    with open(f'{output_dir}/eval/performance_metrics.txt', 'w') as f:
        f.write("Performance metrics:\n")
        pprint.pprint(performance_metrics, stream=f)
        f.close()
    pickle.dump(performance_metrics, open(f'{output_dir}/eval/performance_metrics.pkl', 'wb'))

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
    # Set the data, output path and minimum support
    dataset = 't25i10d10k' # 't25i10d10k' 'toy'
    output_path = f'./output/{dataset}/t_{SIZE}_s{MIN_SUPPORT}_p{PRIVACY_BUDGET}'
    data_path = f'./data/{dataset}/{dataset}.txt'
    evaluate = True

    # Running the algorithms
    algorithms = ['apriori', 'eclat', 'ldp_apriori', 'ldp_eclat']
    for a in algorithms:
        if not os.path.exists(f'{output_path}/{a}.txt'): # Check if the output file already exists
            print(f"Output file for {a} not found. Running the algorithm...")
            frequent_mining(output_path, data_path, a, MIN_SUPPORT)
        else:
            print(f"Output file for {a} already exists. Skipping the algorithm...")
    
    # Evaluate the algorithms
    if evaluate: eval(output_path, algorithms)
