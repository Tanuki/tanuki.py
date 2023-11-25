class IBloomFilterPersistence:
    def save(self, bloom_filter: "BloomFilter") -> None:
        pass

    def load(self, bloom_filter: "BloomFilter") -> None:
        pass
