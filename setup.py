from setuptools import Command, find_packages, setup

__lib_name__ = "SpatialGlue"
__lib_version__ = "1.0.3"
__description__ = "Integrated analysis of spatial multi-omics with SpatialGlue"
__url__ = "https://github.com/JinmiaoChenLab/SpatialGlue"
__author__ = "Yahui Long"
__author_email__ = "longyh@immunol.a-star.edu.sg"
__license__ = "MIT"
__keywords__ = ["Spatial multi-omics", "Cross-omics integration", "Deep learning", "Graph neural networks", "Dual attention"]
__requires__ = ["requests",]

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["SpatialGlue"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = """SpatialGlue is a novel deep learning method for integrating spatial multi-omics data in a spatially informed manner. It utilizes a cycle graph neural network with a dual-attention mechanism to learn the significance of each modality at cross-omics and intra-omics integration. The method can accurately aggregate cell types or cell states at a higher resolution on different tissue types and technology platforms. Besides, it can provide interpretable insights into cross-modality spatial correlations. SpatialGlue is computationally efficient and it only requires about 5 mins for spatial multi-omics data at single-cell resolution (e.g., Spatial-ATAC-RNA-seq data, ~10,000 spots). """,
    long_description_content_type="text/markdown"
)
