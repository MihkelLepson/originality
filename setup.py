from setuptools import find_packages, setup

def main():
    setup(
        name='originality',
        version='0.0.6',
        description=(
            'Calculating text originality in Python'
        ),
        long_description='Check https://github.com/MihkelLepson/originality/tree/main for more information.',
        long_description_content_type="text/markdown",
        packages=find_packages(
            include=['originality'],
            exclude=['cuda_code']),
        install_requires=[
            'numpy >= 1.20.0'
        ],
        package_data={'': ['cuda_lcs_module.so']},
        include_package_data= True,
        url="https://github.com/MihkelLepson/originality",
        download_url="https://github.com/MihkelLepson/originality",
        author="Mihkel Lepson",
        author_email="mihkel.lepson@gmail.com",
        python_requires=f">={3.9}",
        # PyPI package information.
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Environment :: GPU :: NVIDIA CUDA"
        ],
        license="MIT",
        keywords="NLP",
    )

if __name__ == "__main__":
    main()
