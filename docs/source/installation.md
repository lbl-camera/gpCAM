---
banner: _static/electrical-diagram.jpg
banner_text_color: black
banner_shade: 255
banner_brightness: .5
---

# Installation

To install gpCAM do the following:

1. make sure you have Python >=3.7 installed

2. open a terminal

3. create a python environment: e.g. `python3 -m venv test_env`,
   for conda: `conda create --name my_cool_venv_name python=3.8`

4. activate the environment:
  `source ./test_env/bin/activate`,
   conda: `activate my_cool_venv_name` (Windows) and `source activate my_cool_venv_name` (Mac, Linux)

5. type `pip install gpcam` 

6. if any problems occur, update pip `pip install --upgrade pip`,
   setuptools `pip install --upgrade setuptools` and repeat step 5, 
   or try installing from source: `python -m pip install git+https://github.com/lbl-camera/gpCAM.git`
