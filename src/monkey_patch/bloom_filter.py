import hashlib
import math
import os

import numpy as np
from bitarray import bitarray

class BloomFilter:

    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        self.indices = np.zeros(size, dtype=np.int32)

    def hash_functions(self, string):
        # h1(x)
        hash1 = int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16)
        # h2(x)
        hash2 = int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
        return hash1, hash2

    def lookup(self, string):
        hash1, hash2 = self.hash_functions(string)
        for seed in range(self.hash_count):
            index = (hash1 + seed * hash2) % self.size

            #print(f"Lookup: Seed={seed}, Digest={index}, BitValue={self.bit_array[index]}")
            if self.bit_array[index] == 0:
                return False
        return True

    def add(self, string):
        hash1, hash2 = self.hash_functions(string)
        for seed in range(self.hash_count):
            index = (hash1 + seed * hash2) % self.size
            self.bit_array[index] = 1
            #print(f"Add: Seed={seed}, Digest={index}, BitValue={self.bit_array[index]}")

    def save(self, log_directory):
        bloom_filter_path = os.path.join(log_directory, 'bloom_filter_state.bin')
        # Append 0 bits to make the length a multiple of 8
        while len(self.bit_array) % 8 != 0:
            self.bit_array.append(0)
        with open(bloom_filter_path, 'wb') as f:
            f.write(self.bit_array.tobytes())

    def load(self, log_directory):
        bloom_filter_path = os.path.join(log_directory, 'bloom_filter_state.bin')
        bit_array = bitarray()
        with open(bloom_filter_path, 'rb') as f:
            bit_array.frombytes(f.read())
        # Remove any trailing 0 bits that were added for padding
        #while len(bit_array) > 0 and bit_array[-1] == 0:
        #    bit_array.pop()
        self.bit_array = bit_array
        return self

def optimal_bloom_filter_params(n, p):
    """
    Calculate the optimal bit array size (m) and number of hash functions (k)
    for a Bloom filter.

    n: expected number of items to be stored
    p: acceptable false positive probability

    Returns a tuple (m, k)
    """
    m = - (n * math.log(p)) / (math.log(2) ** 2)
    k = (m / n) * math.log(2)
    return int(math.ceil(m)), int(math.ceil(k))