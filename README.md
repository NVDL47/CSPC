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
---
## PW2 - Lab A: Motion from Tracking Data
**What I built:**
- Read noisy free-fall position data from `freefall.csv` (time, height) in `analysis.py`, then used `np.gradient` twice to compute velocity from position and acceleration from velocity.
- Integrated the acceleration back up with `cumulative_trapezoid` (adding the starting value each time) to recover velocity and then position, and compared the recovered position to the original measurements.
- Plotted position, velocity, and acceleration as three stacked panels sharing the time axis in `motion.png`, with a dashed line at -9.81 on the acceleration panel.
**Result:**
- Mean acceleration: -8.57968750000008 m/s² (expected = -9.81), with a standard deviation of 28.71612572170628 m/s² (min -> -73.32499999999908, max -> 54.47499999999985).
- Largest difference between the recovered position and the original: ___ m.
**Noise observation:**
- Taking the derivative amplifies the noise. So even though the original data seemed to be smooth, obtained acceleration was noisy.
**What integrating back showed:**
- Integrating back somewhat cancels out the noise. It smoothes out the noise.
**Bonus (2D trajectory):** Read trajectory.csv and plotted the 2D path (x vs y), which looked like an infinity symbol that is squished. Recorded velocity almost makes a rectangle.
**Conclusion:**
- Everything worked as expected. I learned how to differentiate and integrate with NumPy and SciPy.