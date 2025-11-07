def define_targets(rules):
    rules.py_library(
        name = "torchgen",
        srcs = rules.glob(["**/*.py"]),
        visibility = ["//visibility:public"],
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
        ],
    )

    rules.py_binary(
        name = "gen",
        srcs = [":torchgen"],
        visibility = ["//visibility:public"],
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
        ],
    )
<<<<<<< HEAD

    rules.py_binary(
        name = "gen_executorch",
        srcs = [":torchgen"],
        visibility = ["//visibility:public"],
        deps = [
            rules.requirement("PyYAML"),
            rules.requirement("typing-extensions"),
        ],
    )
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
