"""Install Compacter."""
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def setup_package():
    long_description = (
        "PyTorch CUDA kernel implementation of intrinsic dimension operation."
    )

    setuptools.setup(
        name="seq2seq",
        version="0.0.1",
        description="Compacter",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Rabeeh Karimi Mahabadi",
        license="MIT License",
        packages=setuptools.find_packages(
            exclude=["docs", "tests", "scripts", "examples"]
        ),
        dependency_links=[
            "https://download.pytorch.org/whl/torch_stable.html",
        ],
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7.10",
        ],
        keywords="text nlp machinelearning",
        ext_modules=[
            CUDAExtension(
                "projections.fwh_cuda",
                sources=[
                    "projections/fwh_cuda/fwh_cpp.cpp",
                    "projections/fwh_cuda/fwh_cu.cu",
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        install_requires=[
            "torch==1.8.0+cu111",
        ],
    )


if __name__ == "__main__":
    setup_package()
