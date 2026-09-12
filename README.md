# CSPC - Computer Science for Physics and Chemistry
My coursework repository. Each practical is under PW<n>/Lab <X>/.
## Setup
Create the environment for a given lab:
	conda env create -f PW<n>/Lab\ <X>/environment.yml
	conda activate cspc
---
## PW1 - Lab A: Reproducible Foundations
**What I built:**
- Created a reproducible conda enviroment, made a .gitignore file, and also added a a file called speed.py to measure how much NumPy is faster.
**Speed comparison (loop vs NumPy):**
- loop : 2.892582396999387 s
- numpy : 0.0002968560002045706 s
- speed-up: 9744.059055589374 x faster
**Tests:** all passing: yes
**Conclusion:**
- Everything worked as expected. I learned how to use git branches, and merging them to create a digital lab environment, and also how to use NumPy to have more efficient codes in the future. Faced no major problem.
