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
**Noise observation:**
- Taking the derivative amplifies the noise. So even though the original data seemed to be smooth, obtained acceleration was noisy.
**What integrating back showed:**
- Integrating back somewhat cancels out the noise. It smoothes out the noise.
**Bonus (2D trajectory):** Read trajectory.csv and plotted the 2D path (x vs y), which looked like an infinity symbol that is squished. Recorded velocity almost makes a rectangle.
**Conclusion:**
- Everything worked as expected. I learned how to differentiate and integrate with NumPy and SciPy.
---
## PW2 - Lab B: Optimization in Chemistry
**What I built:**
- In `warmup.py`, minimised the easy function f(x) = (x-3)² + 1 and the harder function g(x) = x⁴ - 3x² + x + 5 with three methods: gradient descent by hand, `newton` applied to the derivative, and `minimize` with SLSQP. For g, I ran each from x0 = 0 and x0 = 2, and checked the sign of g'' at Newton's answer.
- In `kinetics.py`, read `kinetics.csv` (time, concentration), wrote `total_error(k)` as the sum of squared differences from C0·e^(-kt), and minimised it with SLSQP (bounds 0 to 5, start 0.5). Plotted the measured points with the fitted curve in `kinetics.png`.
- In `equilibrium.py`, wrote `k_imbalance(x)` for H2 + I2 ⇌ 2 HI (K = 15.6) and solved it two ways: `newton` root-finding, and SLSQP minimising `k_imbalance(x)**2`. Plotted H2, I2 and HI against the extent x, with the equilibrium marked, in `equilibrium.png`.
- (Bonus) In `titration.py`, read `titration.csv` (volume, pH), computed the slope with `np.gradient`, and found its maximum with `np.argmax`. Plotted the pH curve and its slope side by side in `titration.png`.
**Part 2 - Comparing the methods:**
- On f(x) = (x-3)² + 1, all three methods reached x approx. 3 from x0 = 0. The function is convex, so they agree easily.
- On g(x), the methods disagreed. From x0 = 0, gradient descent and SLSQP reached the global minimum at x approx. -1.307 (g approx. 1.49). Newton instead reached x approx. 0.170, where g'' < 0, so it is a maximum, not a minimum.
- From x0 = 2, all three methods reached the local minimum at x approx. 1.137 (g approx. 3.93), which is higher than the global minimum.
- So the starting point decides which stationary point is found, and Newton only finds a point where g' = 0, which may be a maximum.
**Part 3 - Reaction rate:**
- Fitted rate constant: k = 0.2617613705507395 (expected approx. 0.25). The fitted curve passes through the measured points.
**Part 4 - Equilibrium:**
- Newton gave x = 0.6638476669609822 and SLSQP gave x = 0.6638476594110035. They agree (x is approximately 0.664).
- Equilibrium amounts: H2 = I2 = 0.3361523330390178 mol (approx. 0.336) and HI = 1.3276953339219644 mol (approx. 1.328).
- In the plot, the reactants fall and the product rises as x increases, and the equilibrium line sits at x approx. 0.664.
**Bonus - Titration:**
- Equivalence point: 50.0 mL, where the pH curve jumps and its slope peaks.
**Conclusion:**
- Everything worked as expected. Gradient descent, Newton and SLSQP all work on an easy problem, but on a complicated landscape the start point and the algorithm matter. I also saw that optimisation shows up across chemistry: curve fitting, solving for equilibrium, and finding where a curve is steepest.
---
## PW3 - Data, Distributions, Testing, and Causation
**What I built:**
- In `mechanics.py`, loaded `heart.csv` (Cleveland, UCI/Kaggle) with pandas, plotted the distributions of `age`, `chol`, `trestbps` and `thalach` in `distributions.png`, and checked normality with Q-Q plots (`qqplots.png`) and the Shapiro-Wilk test.
- In `heart_tests.py`, compared `thalach` between disease and healthy patients, computed the mean plus-mius uncertainty for each group (`thalach_means.png`), and measured the age/`thalach` relationship (`age_thalach.png`).
- In `chemicals.py`, correlated benzene and cadmium with malignancy, then repeated it within a narrow `pollution_index` band and as a partial correlation (`chemicals.png`).
**Normality verdicts (Session 1):**
- age: approximately normal (skew = -0.21, excess kurtosis = -0.53, Shapiro p = 0.006)
- chol: not normal (skew = 1.13, excess kurtosis = 4.40, Shapiro p = 5.9e-09)
- trestbps: not normal (skew = 0.70, excess kurtosis = 0.85, Shapiro p = 1.8e-06)
- thalach: not normal, mildly left-skewed (skew = -0.53, excess kurtosis = -0.07, Shapiro p = 7.0e-05)
- Reasoning: Shapiro-Wilk rejects normality for all four, but with 303 patients it also flags tiny deviations. For age the skew and kurtosis are small, so I treat it as roughly normal. chol is strongly right-skewed with heavy tails (max 564), and trestbps is right-skewed. thalach is clearly but moderately skewed to the left.
**Heart rate comparison:**
- Mean thalach: disease = 139.3 (plus-mius 1.9 SEM), healthy = 158.4 (plus-minus 1.5 SEM).
- Test used: Mann-Whitney U, because thalach is not normal. Result: U = 5806.5, p = 1.9e-13. Welch t-test as a cross-check: t = -7.86, p = 9.1e-14.
- Conclusion: the groups differ. Disease patients reach about 19 bpm lower maximum heart rate. The 95% CIs (135.5-143.0 vs 155.4-161.3) do not overlap, which agrees with the test.
**Age vs heart rate:**
- Pearson r = -0.39, Spearman rho = -0.39. Older patients tend to have a lower maximum heart rate (a moderate negative relationship).
**Chemical mystery:**
- Naive: benzene r = 0.38, cadmium r = 0.88, so cadmium appears to be the cause.
- Cadmium is almost perfectly correlated with `pollution_index` (r = 0.98), while benzene is not (r = 0.05).
- After restricting `pollution_index` to 45-55: benzene r = 0.66, cadmium r = -0.04. With pollution removed by regression, benzene r = 0.75 and cadmium r = 0.01.
- Pollution drives cadmium, so cadmium's association with malignancy was a confounding effect. The real cause is benzene, because its association survives when pollution is held fixed, while cadmium's drops to about 0.
**Bonus:**
- Entropy of `target` = 0.995 bits (max 1), with 54% healthy and 46% disease, so the column is almost perfectly balanced and hard to guess.
**Conclusion:**
- Everything worked as expected. Also, Correlation is not causation. And, the test must match the data.