import sys
import importlib
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader


class AliasLoader(Loader):
    def __init__(self, actual_name):
        self.actual_name = actual_name

    def create_module(self, spec):
        return None  # Use default module creation semantics

    def exec_module(self, module):
        actual_module = importlib.import_module(self.actual_name)
        module.__dict__.update(actual_module.__dict__)


class AliasFinder(MetaPathFinder):
    def __init__(self, alias_map):
        self.alias_map = alias_map

    def find_spec(self, fullname, path, target=None):
        if fullname in self.alias_map:
            actual_name = self.alias_map[fullname]
            return spec_from_loader(fullname, AliasLoader(actual_name))
        return None


alias_map = {
    'PyEMD': 'pyemd'
}

# Insert the custom finder at the beginning of sys.meta_path
sys.meta_path.insert(0, AliasFinder(alias_map))

# Verify aliased import functionality
import pyemd as PyEMD
from PyEMD.CEEMDAN import CEEMDAN  # noqa
