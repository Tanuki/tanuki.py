from main import match_email, wrap_match_email


NAMES = [
    "John Smith",
    "Emily Johnson",
    "Michael Davis",
    "Sarah Wilson",
    "Sarah Miller",
    "David Anderson",
    "Jennifer Brown",
    "Robert Lee",
    "Linda Taylor",
    "William Jones",
    "Mary Jackson",
    "James White",
    "Patricia Martinez",
    "Daniel Wilson",
    "Susan Clark",
    "Joseph Garcia",
    "Karen Harris",
    "Richard Miller",
    "Nancy Thomas",
    "Charles Lewis",
    "Jennifer Allen",
    "Thomas Walker",
    "Margaret Wright",
    "Christopher Scott",
    "Betty Hall",
    "Daniel Moore",
    "Dorothy King",
    "Matthew Turner",
    "Lisa Green",
    "Donald Baker",
    "Helen Adams",
    "Mark Turner",
    "Nancy Collins",
    "Paul Davis",
    "Sharon Edwards",
    "George Taylor",
    "Cynthia Martinez",
    "Kenneth Jackson",
    "Angela Clark",
    "Steven Allen",
    "Donna Miller",
    "Edward Harris",
    "Ruth Wilson",
    "Brian Young",
    "Deborah Martin",
    "Ronald Anderson",
    "Jessica Thompson",
    "Gary Hernandez",
    "Lisa Davis",
    "Jerry Parker",
    "Kimberly Johnson",
    "Sarah Edward",
]


def test_match_email() -> None:
    assert match_email("ava.anderson@cool.com", ["Ava Anderson"]) == ["Ava Anderson"]
    assert match_email("aanderson@cool.com", ["Ava Anderson", "Alan Anderson"]) == ["Ava Anderson", "Alan Anderson"]


def test_match_email_none() -> None:
    assert not match_email("bob@cool.ai", ["Emily Johnson"])


def test_wrap_match_email() -> None:
    assert wrap_match_email("eharris@example.ai", NAMES) == "Edward Harris"
    assert wrap_match_email("sedward@example.com", NAMES) == "Sarah Edward"
    assert wrap_match_email("l.davis@example.com", NAMES) == "Lisa Davis"

    assert not wrap_match_email("davis@example.com", NAMES)
    assert not wrap_match_email("sarah@test.com", NAMES)
