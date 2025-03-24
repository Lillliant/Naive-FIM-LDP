def process_data(input_path: str) -> list[list[int]]:
    data = []
    with open(input_path, 'r') as f:
        line = f.readline()
        data.append(list(map(int, line.strip().split())))
    return data

if __name__ == '__main__':
    process_data('./data/t25i10d10k/t25i10d10k.txt')