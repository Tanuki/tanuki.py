from tanuki.validator import Validator


def test_validate_base_type():
    print("test_validate_base_type")
    validator = Validator()
    assert validator.validate_base_type(1, int)
    assert validator.validate_base_type(1.0, float)
    assert validator.validate_base_type("1", str)
    assert validator.validate_base_type(True, bool)
    assert validator.validate_base_type(None, None)
    assert not validator.validate_base_type(1, float)
    assert not validator.validate_base_type(1.0, int)
    assert not validator.validate_base_type("1", int)
    #assert not validator.validate_base_type(True, int)
    assert not validator.validate_base_type(None, int)
    assert not validator.validate_base_type(1, str)
    assert not validator.validate_base_type(1.0, str)
    assert not validator.validate_base_type(True, str)
    assert not validator.validate_base_type(None, str)
    assert not validator.validate_base_type(1, bool)
    assert not validator.validate_base_type(1.0, bool)
    assert not validator.validate_base_type("1", bool)
    assert not validator.validate_base_type(None, bool)
    assert not validator.validate_base_type(1, None)
    assert not validator.validate_base_type(1.0, None)
    assert not validator.validate_base_type("1", None)
    assert not validator.validate_base_type(True, None)

if __name__ == "__main__":
    test_validate_base_type()