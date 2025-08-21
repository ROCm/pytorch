from setuptools import setup, find_packages

setup(
    name='TraceLens',
    version='0.1.0',
    packages=find_packages(where='.'),  # Will pick up 'TraceLens' automatically
    package_dir={"": "."},
    install_requires=[
        'pandas',
        'tqdm',
        'backports.strenum;python_version<"3.11"',
        'StrEnum;python_version<"3.11"',
        # 'openpyxl',
        # 'tensorboard-plugin-profile==2.19.0',
        # 'tensorflow',
    ],
    description="A library for Automating analysis from PyTorch trace files",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AMD-AIG-AIMA/TraceLens', 
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": [
            "TraceLens_jax_analysis_perf_reporting = TraceLens.Reporting.generate_perf_report_jax_analysis:main",
            "TraceLens_generate_perf_report_pytorch = TraceLens.Reporting.generate_perf_report_pytorch:main",
            "TraceLens_compare_perf_reports_pytorch = TraceLens.Reporting.compare_perf_reports_pytorch:main",
        ],
    },
)
