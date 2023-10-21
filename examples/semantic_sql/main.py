import dis

from monkey import Monkey as monkey


@monkey.patch
def semantic_sql(input: str, schema: str) -> str:
    """
    Convert the input into a valid SQL query. Assume that the database has a table named "user" with
    columns "id", "organization_id", "name", and "email_notify"
    """

@monkey.patch
def semantic_sql_var(input: str, table_name: str = None, columns=[]) -> str:
    """
    Convert the input into a valid SQL query. Assume that the database has a table named "user" with
    columns "id", "organization_id", "name", and "email_notify"
    """


@monkey.align
def test_semantic_sql_var():
    output = semantic_sql_var("list the names of all users in our user database", table_name="user", columns=["id", "organization_id", "name", "email_notify"])
    assert output == "SELECT name FROM user"

@monkey.align
def test_semantic_sql():
    output = semantic_sql(input = "list the names of all users in our user database", schema = """table_name=user, columns=["id", "organization_id", "name", "email_notify"]""")
    assert output == "SELECT name FROM user"


if __name__ == '__main__':
    test_semantic_sql_var()
    #test_semantic_sql()
    print("Everything passed")