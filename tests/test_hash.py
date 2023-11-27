from tanuki.bloom_filter import BloomFilter
from tanuki.trackers.filesystem_buffered_logger import FilesystemBufferedLogger


def test_hash():
    logger = FilesystemBufferedLogger("test")
    bloom_filter_persistence = logger.get_bloom_filter_persistence()
    bloom = BloomFilter(bloom_filter_persistence, expected_number_of_elements=100, false_positive_probability=0.01)
    assert bloom.hash_functions("test") == bloom.hash_functions("test")

if __name__ == "__main__":
    test_hash()