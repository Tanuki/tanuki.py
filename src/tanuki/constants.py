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

# default models
DEFAULT_TEACHER_MODEL_NAMES = ["gpt-4", "gpt-4-32k", ]
DEFAULT_DISTILLED_MODEL_NAME = "gpt-3.5-turbo-1106"
DEFAULT_EMBEDDING_MODEL_NAME = "ada-002"

# provider names
OPENAI_PROVIDER = "openai"
LLAMA_BEDROCK_PROVIDER = "llama_bedrock"
TITAN_BEDROCK_PROVIDER = "aws_titan_bedrock"
TOGETHER_AI_PROVIDER = "together_ai"

# model type strings
TEACHER_MODEL = "teacher"
DISTILLED_MODEL = "distillation"