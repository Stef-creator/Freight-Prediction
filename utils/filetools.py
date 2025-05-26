import chardet

def find_encoding(filepath):
    """
    Detect and print the likely encoding of a file.

    Args:
        filepath (str): Path to the file to check.

    Returns:
        dict: Encoding metadata, e.g. {'encoding': 'utf-16', 'confidence': 0.99}
    """
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read(100000))
        print(result)
        return result
