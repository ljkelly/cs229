# CS229 Machine Learning Project
Paavani Dua (pavanid)
Liam Kelly (kellylj)
Matthew Lee (mattskl)

## Setup for Coding Parts

1. Install [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. cd into code, run `conda env create -f environment.yml`
  - This creates a Conda environment called `cs229-proj`
3. Run `source activate cs229-proj`
  - This activates the `cs229-proj` environment
  - Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/cs229-proj/bin/python`
  - Select `OK` then `Apply`
