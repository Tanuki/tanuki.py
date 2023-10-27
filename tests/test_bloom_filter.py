import os

from bloom_filter import BloomFilter, optimal_bloom_filter_params
from models.function_example import FunctionExample
from trackers.buffered_logger import BufferedLogger, EXPECTED_ITEMS, FALSE_POSITIVE_RATE


def test_add():
    #bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    logger = BufferedLogger("test")
    example = FunctionExample((0,), {}, 0 * 2)
    example_data = str(example.__dict__).encode('utf-8') + b'\n'

    before_bit_array = logger.bloom_filter.bit_array.copy()

    logger.log_patch("test", example)

    after_bit_array = logger.bloom_filter.bit_array
    is_same = before_bit_array == after_bit_array

    assert is_same == False

    looked_up = logger.bloom_filter.lookup("test_" + example_data.decode('utf-8'))
    assert looked_up == True

def test_add2():
    logger = BufferedLogger("test")
    example = FunctionExample((0,), {}, 0 * 2)
    print(id(logger.bloom_filter))
    logger.log_patch("test", example)
    print(id(logger.bloom_filter))

    looked_up = logger.bloom_filter.lookup("test_" + str(example.__dict__)+"\n")
    assert looked_up == True

def test_add_lookup():
    bloom_filter = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    example = FunctionExample((0,), {}, 0 * 2)

    bloom_filter.add(str(example.__dict__))
    looked_up = bloom_filter.lookup(str(example.__dict__))

    assert looked_up == True


def test_bloom_filter_persistence():
    # Create a Bloom filter and add some values
    bf1 = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    for i in range(10):
        bf1.add(str(i))

    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Save the Bloom filter's state
    bf1.save(current_dir)
    saved_bytes = bf1.bit_array.tobytes()

    assert bf1.bit_array.count(1) != 0
    num = 50

    # Manual saving
    with open("test.bin", "wb") as f:
        f.write(saved_bytes)

    # Manual loading
    with open("test.bin", "rb") as f:
        manual_loaded_bytes = f.read()

    assert saved_bytes == manual_loaded_bytes


    # Create a new Bloom filter and load its state
    bf2 = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    bf2.load(current_dir)
    loaded_bytes = bf2.bit_array.tobytes()

    assert saved_bytes == loaded_bytes

    for i, (b1, b2) in enumerate(zip(saved_bytes, loaded_bytes)):
        if b1 != b2:
            print(f"Discrepancy at byte {i}: saved = {b1}, loaded = {b2}")

    # Check if the two Bloom filters are equal
    assert saved_bytes == loaded_bytes
    # Bit wise comparison of the two Bloom filters
    for i in range(len(bf1.bit_array)):
        if not bf1.bit_array[i] == bf2.bit_array[i]:
            print(i)
            print(bf1.bit_array[i-10:i+10])
            print(bf2.bit_array[i-10:i+10])
        assert bf1.bit_array[i] == bf2.bit_array[i]

    assert bf1.bit_array == bf2.bit_array

    # Check if the loaded Bloom filter behaves as expected
    for i in range(10):
        assert bf2.lookup(str(i))

    print("Test passed!")


def test_bit_array_length():
    bf1 = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    current_dir = os.path.dirname(os.path.realpath(__file__))
    bf1.save(current_dir)

    # Print length of bit_array before loading
    print("Length of bit_array before loading:", len(bf1.bit_array))

    # Print size of the saved file
    file_size = os.path.getsize(os.path.join(current_dir, 'bloom_filter_state.bin'))
    print("Size of saved file (in bytes):", file_size)

    bf2 = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    bf2.load(current_dir)

    # Print length of bit_array after loading
    print("Length of bit_array after loading:", len(bf2.bit_array))

    assert len(bf1.bit_array) == len(bf2.bit_array), "Mismatch in bit array lengths."


def test_file_content_consistency():
    bf1 = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    current_dir = os.path.dirname(os.path.realpath(__file__))
    bf1.save(current_dir)

    # Read back immediately after saving
    with open(os.path.join(current_dir, 'bloom_filter_state.bin'), 'rb') as f:
        saved_data = f.read()

    bf2 = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    bf2.load(current_dir)

    assert saved_data == bf2.bit_array.tobytes(), "Saved file content doesn't match loaded content."


def test_simple_test_case():
    bf1 = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    items = ['test1', 'test2', 'test3']

    for item in items:
        bf1.add(item)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    bf1.save(current_dir)

    bf2 = BloomFilter(*optimal_bloom_filter_params(EXPECTED_ITEMS, FALSE_POSITIVE_RATE))
    bf2.load(current_dir)

    for item in items:
        assert bf2.lookup(item), f"Item {item} not found after loading."



if __name__ == "__main__":
    test_bit_array_length()
    test_file_content_consistency()
    test_bloom_filter_persistence()
    test_add2()
    test_bloom_filter_persistence()
    #test_add_lookup()
    test_add()
