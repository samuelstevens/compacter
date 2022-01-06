"""Install Compacter."""
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

description = "PyTorch CUDA kernel implementation of intrinsic dimension operation."


def setup_package():

    setuptools.setup(
        name="intrinsic",
        version="0.0.1",
        description=description,
        long_description=description,
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
            "Programming Language :: Python :: 3.9.7",
        ],
        keywords="text nlp machinelearning",
        ext_modules=[
            CUDAExtension(
                "intrinsic.fwh_cuda",
                sources=[
                    "intrinsic/fwh_cuda/fwh_cpp.cpp",
                    "intrinsic/fwh_cuda/fwh_cu.cu",
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
