import os

from bitarray._bitarray import bitarray

from monkey_patch.bloom_filter import BloomFilter
from monkey_patch.persistence.filter.bloom_interface import IBloomFilterPersistence


class FileSystemBloomFilterPersistence(IBloomFilterPersistence):
    """
    This is a Filesystem implementation of a Bloom Filter persistence layer.
    """

    def __init__(self, log_directory: str):
        self.log_directory = log_directory

    def save(self, bloom_filter: BloomFilter):
        """
        Write a bloom filter array of bits to the local filesystem.
        :param bloom_filter: A bloom filter which tracks unique function invocations
        """
        bloom_filter_path = os.path.join(self.log_directory, 'bloom_filter_state.bin')
        # Append 0 bits to make the length a multiple of 8
        while len(bloom_filter.bit_array) % 8 != 0:
            bloom_filter.bit_array.append(0)
        with open(bloom_filter_path, 'wb') as f:
            f.write(bloom_filter.bit_array.tobytes())

    def load(self) -> BloomFilter:
        """
        Load a bloom filter from the local filesystem.
        :return: A bloom filter object containing the state of unique function invocations
        """
        bloom_filter_path = os.path.join(self.log_directory, 'bloom_filter_state.bin')
        with open(bloom_filter_path, 'rb') as f:
            bit_array = bitarray()
            bit_array.frombytes(f.read())
        # Remove any trailing 0 bits that were added for padding
        #while len(bit_array) > 0 and bit_array[-1] == 0:
        #    bit_array.pop()
        self.bit_array = bit_array
        return self