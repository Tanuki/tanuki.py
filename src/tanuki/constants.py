from typing import Literal

EXAMPLE_ELEMENT_LIMIT = 1000

# These represent the file extensions for the symbolic patch and alignment datasets
PATCHES = "patches"
PATCH_FILE_EXTENSION_TYPE = Literal[".patches"]
PATCH_FILE_EXTENSION: PATCH_FILE_EXTENSION_TYPE = f".patches"

SYMBOLIC_ALIGNMENTS = "alignments"
ALIGN_FILE_EXTENSION_TYPE = Literal[".alignments"]
ALIGN_FILE_EXTENSION: ALIGN_FILE_EXTENSION_TYPE = f".alignments"

# These represent the file extensions for the embeddable examples positive and negative datasets
POSITIVE_EMBEDDABLE_ALIGNMENTS = "positive"
POSITIVE_FILE_EXTENSION_TYPE = Literal[".positive"]
POSITIVE_FILE_EXTENSION: POSITIVE_FILE_EXTENSION_TYPE = ".positive"

NEGATIVE_EMBEDDABLE_ALIGNMENTS = "negative"
NEGATIVE_FILE_EXTENSION_TYPE = Literal[".negative"]
NEGATIVE_FILE_EXTENSION: NEGATIVE_FILE_EXTENSION_TYPE = ".negative"

# Bloom filter default config
EXPECTED_ITEMS = 10000
FALSE_POSITIVE_RATE = 0.01

# The name of the library
LIB_NAME = "tanuki"
ENVVAR = "TANUKI_LOG_DIR"