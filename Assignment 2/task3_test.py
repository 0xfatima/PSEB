import pytest
from task3main import is_adult


def test_underage():
    assert is_adult(17) ==False
    
def test_adult():
    assert is_adult(19) == True
    
def test_negative():
    with pytest.raises(ValueError):
        is_adult(-1)
