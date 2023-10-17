import hashlib
from bitarray import bitarray


class BloomFilter:

    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, string):
        for seed in range(self.hash_count):
            result = hashlib.sha256(string.encode('utf-8')).hexdigest()
            digest = int(result, 16)
            index = (seed * 2 + digest) % self.size
            self.bit_array[index] = 1

    def lookup(self, string):
        for seed in range(self.hash_count):
            result = hashlib.sha256(string.encode('utf-8')).hexdigest()
            digest = int(result, 16)
            index = (seed * 2 + digest) % self.size
            if self.bit_array[index] == 0:
                return False
        return True
