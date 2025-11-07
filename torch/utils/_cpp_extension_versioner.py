# mypy: allow-untyped-defs
import collections


<<<<<<< HEAD
Entry = collections.namedtuple('Entry', 'version, hash')
=======
Entry = collections.namedtuple("Entry", "version, hash")
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


def update_hash(seed, value):
    # Good old boost::hash_combine
    # https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
<<<<<<< HEAD
    return seed ^ (hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2))
=======
    return seed ^ (hash(value) + 0x9E3779B9 + (seed << 6) + (seed >> 2))
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


def hash_source_files(hash_value, source_files):
    for filename in source_files:
<<<<<<< HEAD
        with open(filename, 'rb') as file:
=======
        with open(filename, "rb") as file:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            hash_value = update_hash(hash_value, file.read())
    return hash_value


def hash_build_arguments(hash_value, build_arguments):
    for group in build_arguments:
        if group:
            for argument in group:
                hash_value = update_hash(hash_value, argument)
    return hash_value


class ExtensionVersioner:
    def __init__(self):
        self.entries = {}

    def get_version(self, name):
        entry = self.entries.get(name)
        return None if entry is None else entry.version

<<<<<<< HEAD
    def bump_version_if_changed(self,
                                name,
                                source_files,
                                build_arguments,
                                build_directory,
                                with_cuda,
                                with_sycl,
                                is_python_module,
                                is_standalone):
=======
    def bump_version_if_changed(
        self,
        name,
        source_files,
        build_arguments,
        build_directory,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone,
    ):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        hash_value = 0
        hash_value = hash_source_files(hash_value, source_files)
        hash_value = hash_build_arguments(hash_value, build_arguments)
        hash_value = update_hash(hash_value, build_directory)
        hash_value = update_hash(hash_value, with_cuda)
        hash_value = update_hash(hash_value, with_sycl)
        hash_value = update_hash(hash_value, is_python_module)
        hash_value = update_hash(hash_value, is_standalone)

        entry = self.entries.get(name)
        if entry is None:
            self.entries[name] = entry = Entry(0, hash_value)
        elif hash_value != entry.hash:
            self.entries[name] = entry = Entry(entry.version + 1, hash_value)

        return entry.version
