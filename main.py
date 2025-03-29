import datetime
from algorithms.apriori import apriori
from algorithms.eclat import eclat
import os

def process_data(input_path: str) -> list[list[int]]:
    data = []
    with open(input_path, 'r') as f:
        while line := f.readline():
            data.append(list(map(int, line.strip().split())))
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

def frequent_mining(output_path: str, data_path: str, min_support: int):
    # Read the input data
    print("Reading input data...")
    data = process_data(data_path)

    # Processing for Apriori
    print("Processing with Apriori...")
    a_stime = datetime.datetime.now()
    print("Start time: ", a_stime)
    frequent_itemset = apriori(data, min_support)
    a_etime = datetime.datetime.now()
    print("End time: ", a_etime)
    write_output(output_path, 'apriori', frequent_itemset, a_stime, a_etime)

    # Processing for Eclat
    print("Processing with Eclat...")
    e_stime = datetime.datetime.now()
    print("Start time: ", e_stime)
    frequent_itemset = eclat(data, min_support)
    e_etime = datetime.datetime.now()
    print("End time: ", e_etime)
    write_output(output_path, 'eclat', frequent_itemset, e_stime, e_etime)

if __name__ == '__main__':
    # Set the output path and minimum support
    min_support = 900
    output_path = f'./output/original_9976_s{min_support}/'
    data_path = './data/t25i10d10k/t25i10d10k.txt'
    
    frequent_mining(output_path, data_path, min_support)
