from bloom_filter import BloomFilter


def test_hash():
    bloom = BloomFilter(100, 0.01)
    assert bloom.hash_functions("test") == bloom.hash_functions("test")

if __name__ == "__main__":
    test_hash()