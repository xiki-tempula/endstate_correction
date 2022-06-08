"""
Unit and regression test for the endstate_rew package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import endstate_rew


def test_endstate_rew_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "endstate_rew" in sys.modules


def test_hipen_import():
    
    from endstate_rew.system import _get_hipen_data
    print(_get_hipen_data())
