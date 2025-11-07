def define_targets(rules):
    rules.py_library(
        name = "autograd",
        srcs = rules.glob(["*.py"]),
        data = rules.glob([
            "*.yaml",
            "templates/*",
        ]),
        visibility = ["//:__subpackages__"],
        deps = [
            rules.requirement("PyYAML"),
            "//torchgen",
        ],
    )
<<<<<<< HEAD

    rules.filegroup(
        name = "deprecated_yaml",
        srcs = ["deprecated.yaml"],
        visibility = ["//:__subpackages__"],
    )
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
