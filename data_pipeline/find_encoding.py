import chardet


def find_encoding(filepath):
    with open(f'{filepath}', 'rb') as f:
        result = chardet.detect(f.read(100000))
        print(result)

