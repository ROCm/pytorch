def define_targets(rules):
    rules.py_binary(
        name = "generate_code",
        srcs = ["generate_code.py"],
        visibility = ["//:__pkg__"],
        deps = [
            rules.requirement("PyYAML"),
            "//tools/autograd",
            "//torchgen",
        ],
    )

    rules.py_binary(
        name = "gen_version_header",
        srcs = ["gen_version_header.py"],
<<<<<<< HEAD
        visibility = [
            "//:__pkg__",
            "//torch/headeronly:__pkg__",
        ],
=======
        visibility = ["//:__pkg__"],
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    )
