from setuptools import setup, find_packages

setup(name="combat-stadis",
      version="0.0.0",
      description="Compare observed and theoretical statistical distibutions",
      author="Athanaseus Ramaila",
      author_email="aramaila@ska.ac.za",
      packages=find_packages(),
      url="https://github.com/Athanaseus/combat-stadis",
      license="GNU GPL 3",
      classifiers=["Intended Audience :: Developers",
                   "Programming Language :: Python :: 2",
                   "Topic :: Software Development :: Libraries :: Python Modules"],
      platforms=["OS Independent"],
      install_requires=["aimfast",
                        "tabletext",
                        "matplotlib==2.2.3"],
      extras_require={'docs': ["sphinx-pypi-upload",
                               "numpydoc",
                               "Sphinx"]},
      scripts=['combat/bin/combat'])
