from rlcourse.system_setup import setup_ignore_warnings

def pytest_configure(config):
    # Ignore known deprecation warnings from pygame/pkg_resources
    setup_ignore_warnings()
