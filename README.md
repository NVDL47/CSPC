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
---
## PW1 - Lab B: Data, Plotting, and Automation
**What I built:**
- Read observed radioactive decay data from `decay_observed.csv` and compared it against the analytical decay law `N(t) = N0 * exp(-LAMBDA * t)`, using `LAMBDA = 0.3` and the first observed count as `N0`. Plotted both side by side in `figure.png` using a shared-axis subplot in `plot.py`.
- Automated the figure generation with a `Snakefile`, so that running `snakemake --cores 1 figure.png` regenerates `figure.png` from `decay_observed.csv` only when needed, using `plot.py`.

**Result:**
- The observed data follows the same overall exponential decay trend as the analytical curve, although the observed values do not match it exactly.

**Tests:** Snakemake correctly skipped re-running when no inputs changed, and rebuilt `figure.png` when the file was deleted: yes

**Conclusion:**
- Everything worked as expected. I learned how to read CSV data with NumPy, build comparison plots with Matplotlib, and use Snakemake to automate a small pipeline so outputs stay in sync with their inputs. Faced no major problem.
