# Spine Registration

This repository contains several Python scripts and example notebooks for registering a microscopy image of spinal tissue to an atlas image of the same location. Documentation for every function within this package can be found here: https://twardlab.github.io/spine_registration.

The pipeline contains X key stages:
(1) Load + preprocess input files (I, J, L, pointsJ)
- Include images of I, H, and L before (and after, if applicable) preprocessing
(2) Register the target image, J,  to the atlas image, I
- Include images of J before and after registration
- Also include important plots
(3) Postprocess + save relevant outputs
- Define all outputs

Final contents:
- spine_reg.py
- spine_reg_pipeline.py
- example.ipynb

TODO
- Remove old notebook after docs and rst of repo are finished
