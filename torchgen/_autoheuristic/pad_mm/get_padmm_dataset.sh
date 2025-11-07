#!/bin/bash

a100_zip="pad_mm_a100_data.zip"
<<<<<<< HEAD
a100_data="https://github.com/AlnisM/autoheuristic-datasets/raw/main/${a100_zip}"
=======
a100_data="https://github.com/AlnisM/autoheuristic-datasets/raw/main/${a100_zip}"  # @lint-ignore
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
rm -f ${a100_zip}
wget ${a100_data}
unzip -o ${a100_zip}
rm ${a100_zip}
