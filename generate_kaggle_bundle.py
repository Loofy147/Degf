import json

def get_file_content(path):
    with open(path, "r") as f:
        return f.read()

files = ["degf_core.py", "monitor_gpt2.py", "degf_v6.py", "sgs2_prototype.py", "train_thermo.py", "ablation_a3.py", "hallucination_protocol.py"]

setup_code = "import os\n"
for f in files:
    content = get_file_content(f)
    setup_code += f"with open('{f}', 'w') as f: f.write({repr(content)})\n"

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DEGF v6: Full Empirical Suite (GPU-Ready)\n",
    "This notebook contains the complete DEGF v6 framework and experiments."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install transformer_lens jaxtyping einops"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    setup_code,
    "print('DEGF modules extracted successfully.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python3 degf_v6.py --real"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("degf_v6_bundle.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Kaggle bundle generated at degf_v6_bundle.ipynb")
