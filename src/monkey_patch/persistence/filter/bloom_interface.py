class IBloomFilterPersistence:
    def save(self, bloom_filter: "BloomFilter"):
        pass

    def load(self) -> "BloomFilter":
        pass
