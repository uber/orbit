from mock import patch

import setuptools  # noqa


@patch('setuptools.setup', return_value=True)
def test_setup(mock_func):

    try:
        from setup import setup
        setup.setup()
        assert True
    except:
        assert False
