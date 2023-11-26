from bitarray import bitarray


class IBloomFilterPersistence:
    def save(self, bit_array: bitarray) -> None:
        pass

    def load(self) -> bitarray:
        pass
