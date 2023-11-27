from tanuki.validator import Validator


def test_base_type():
    print("test_base_type")
    validator = Validator()
    assert validator.is_base_type(int)
    assert validator.is_base_type(float)
    assert validator.is_base_type(str)
    assert validator.is_base_type(bool)
    assert validator.is_base_type(None)
    assert not validator.is_base_type(list)
    assert not validator.is_base_type(dict)
    assert not validator.is_base_type(tuple)
    assert not validator.is_base_type(set)


if __name__ == "__main__":
    test_base_type()