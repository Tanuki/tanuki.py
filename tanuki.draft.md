# Tanuki v0.2.0 - Easily build LLM-powered apps that get cheaper and faster over time.

![Discord](https://img.shields.io/discord/1168948553222197248)

Tanuki (formerly MonkeyPatch) aims to reduce the time to ship your LLM projects, so you can focus on building what your users want, instead of on MLOps. We'll quickly cover 1) what makes using Tanuki so easy for LLM apps and 2) what AI functions are and how to use them effectively in your LLM projects. 

To start off, Tanuki allows you to define AI functions in your code:

```python
@tanuki.patch
def get_sql(schema, instruction: str = "", **kwargs) -> SQLQuery:
    """
    Generates an SQL query for the DB defined by the schema that fulfils the instruction.
    """
```

These functions can be invoked as normal, and used in the rest of your program.

Instead of prompt engineering or fine-tuning models scratch, you can precisely define behaviour of these functions using `align` functions. This is how you declare the intended behaviour using `assert` syntax:

```python
@tanuki.align
def align_sql(schema):

    correct_customer_sql: str = 'SELECT * FROM transactions INNER JOIN customer ON transactions.customer = customer.id WHERE customer.account_number = 6913'
     
    # This align statement defines how `get_sql` should behave
    assert get_sql("Get all customer transactions using their account_number", 
				   schema, 
				   account_number=6531) == correct_customer_sql
```
This is called **test-driven alignment** and is an evolution from how LLMs have traditionally been productionized. This paradigm greatly simplifies the work of developers without conceding control. We'll expand on this below.

## Test-Driven Alignment

These optional align statements allow you to define and develop the LLM's behaviour to fit your requirements. Getting well-typed outputs and accounting for edge cases is now trivially easy. Also, representing behaviour declaratively in your code means that you don't require external datasets or an MLOps process to deploy LLM features. 

This allows you to iterate on a product faster, as incremental improvements to the behaviour of your functions are tracked by version control, and visible to all stakeholders. 

As there is no additional state that you have to worry about, improving the performance of your functions is matter of writing more and better `align` assert statements. In experiments, we've seen anywhere from 8-12 tests to achieve sufficient accuracy for various enterprise use-cases.  

As in [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development), this approach means that the tests define the contract of the function. LLM features can now have the same development lifecycle as the rest of the application. 

# AI Functions

AI Functions are how you integrate Tanuki with the rest of your application. Tanuki supports two different classes of AI functions for different use-cases; namely, *symbolic* and *embeddable*. 
## Symbolic Functions
---
![Static Badge](https://img.shields.io/badge/web_development-blue) ![Static Badge](https://img.shields.io/badge/chat-blue) ![Static Badge](https://img.shields.io/badge/parsing-blue) ![Static Badge](https://img.shields.io/badge/tools-blue) ![Static Badge](https://img.shields.io/badge/sparse_retrieval-blue)
In Tanuki, a symbolic functions is a patched function that returns custom objects or primitives - much like a regular function in Python.

These functions are ideal for tasks where the output is more structured and defined, such as generating data objects, parsing information, or even creating code snippets. 

```python
@tanuki.patch
def parse_invoice(invoice_text: str) -> Optional[Invoice]:
    """
    Parses the given invoice text and returns an Invoice object with structured data.
    """
```

These functions can either be aligned explicitly in code:

```python
@tanuki.align
def align_invoice():

    invoice_text = "Invoice #123: Total $200" 
    expected_invoice = Invoice(number=123, total=200)
     
    # Here is your align statement
    assert parse_invoice(invoice_text) == expected_invoice
```

Or from a ground truth dataset that you already have access to:

```python
@tanuki.align
def align_invoice(invoice_ground_truth_csv_path):
    # Load and iterate through each row in the DataFrame
    for index, row in pd.read_csv(invoice_ground_truth_csv_path)
        # Assuming your CSV has columns named 'query', 'expected_sql', and 'account_number'
        invoice_text = row['invoice_text']
        expected_invoice = Invoice(**json.loads(row['expected_invoice']))
        # Perform the assert for each row
        assert parse_invoice(invoice_text) == expected_invoice
```
## Embeddable Functions (New in v0.1.0!)
![Static Badge](https://img.shields.io/badge/indexing-blue) ![Static Badge](https://img.shields.io/badge/dense_retrieval-blue) ![Static Badge](https://img.shields.io/badge/semantic_search-blue)

New in v0.1.0 you can now create embeddable functions that return vectors. These embeddings are crucial for tasks like semantic search, content recommendation, or any other scenario where you need to index and retrieve information based on similarity. 

```python
@tanuki.patch
def embed_document(news_article: str, **kwargs) -> Embedding[np.ndarray[float]]:
    """
    A news article
    """
```

These functions can be aligned like the previous symbolic functions. You declare which examples should have similar embeddings, and which should have dissimilar embeddings.

```python
@tanuki.align
def align_embeddings():
    # These articles should have a similar representation
    assert embed_document("This is an article about dogs") == embed_document("This is an article about cats")
    # These articles are dissimilar, and so their embeddings should be 'pushed' apart.
    assert embed_document("This is an article about dogs") == embed_document("This is an article about cars")
```

These alignments can be used to contrastively fine-tune your embedding backend. Although currently only OpenAI Ada is supported, more model providers are on the way.

