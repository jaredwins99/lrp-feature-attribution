# Layer-Wise Relevance Propagation for Tabular Data with Graphical Dependencies
**Part of an interpretable machine learning seminar: Seeing whether a deep-learning saliency method can handle features correlated in network patterns**

*STATUS: Pre-LMM*

**Note**: This is just an overview of the project. For detailed analysis, methodologies, and insights, refer to the
main [report](https://github.com/jaredwins99/lrp-feature-attribution/blob/main/IML_Final_Paper__Attribution_for_Data_w
ith_Graphical_Feature_Dependencies.pdf).

---

## File Overview

- [main.ipynb](https://github.com/jaredwins99/lrp-feature-attribution/blob/main/main.ipynb): primary notebook, simulation pipeline, MLP training, LRP attribution, and result visualizations
- [linear.ipynb](https://github.com/jaredwins99/lrp-feature-attribution/blob/main/linear.ipynb): experiments on the linear case (standardization, bias, split-domain ablations)
- [generative_functions.py](https://github.com/jaredwins99/lrp-feature-attribution/blob/main/generative_functions.py): correlation-matrix construction (`Σ = N(WWᵀ + D)N`), eigenvector centrality, and feature simulators
- [inferential_functions.py](https://github.com/jaredwins99/lrp-feature-attribution/blob/main/inferential_functions.py): `BasicRegressionNN` MLP, training loop, and the `lrp` routine implementing the ε / γ / w² rules across layers
- [plotting_functions.py](https://github.com/jaredwins99/lrp-feature-attribution/blob/main/plotting_functions.py): heatmap and correlation-network visualization utilities

<br>
<img width="900" alt="Correlation matrices as a function of max eigenvector centrality and eigenvector centrality
entropy" src="plots/all_correlations.png" />
<br>

---

## Introduction
Layer-wise relevance propagation (LRP) is a post-hoc, model-specific interpretability method that backpropagates a model's output through its activations to assign a relevance score to each input feature. 
It has been used heavily in computer vision to produce saliency maps, but those maps are hard to validate. 
Recent work has shown adversarial inputs can fool a model while leaving its saliency map essentially unchanged.

No work has applied LRP to **tabular data with a known dependence structure**. This project builds complex simulations where the ground-truth feature importances are known by construction. 
Can LRP recover them, and what happens to attribution quality as correlation (measured by eigenvector centrality) between features grows?

---

## Graphical Dependencies
Tabular data is often called "unstructured" because, unlike images or text, it has no spatial or sequential grid. 
Even so, we can *impose* an arbitrary dependence structure and treat the correlation matrix as a weighted adjacency matrix on the feature graph.

That reframing lets graph-theoretic metrics describe the structure:
- **Max eigenvector centrality** — how concentrated the influence is on a single feature
- **Eigenvector-centrality entropy** — how evenly correlation is spread across all features

<br>
<img width="500" alt="Correlation network graph" src="plots/graph_visual.png" />
<br>

The graph above shows one such correlation structure as a network with edge color/opacity encoding pairwise correlation strength.

---

## Simulation Methodology
The data generation is done in three progressively complex stages:
1. **Uncorrelated linear** — 10 standard-normal features, with the first 5 contributing to `y` with various parameters plus 5 noise features
2. **Split** — feature 8 only contributes to odd-indexed observations, while feature 0 only contributes to
even-indexed ones, with an indicator feature added to help the MLP learn the regime
3. **Correlated** — features drawn from a multivariate normal whose covariance is built as `Σ = N(WWᵀ + D)N`,
normalized to a correlation matrix

`W` is taken as an increasing sequence raised to a power (controlling skewness, and thus eigenvector-centrality
entropy), and `D` controls the diagonal magnitude (and thus the maximum achievable correlation). Sweeping these gives
the 5×5 grid of correlation structures shown at the top of this README.

---

## Model and LRP Setup
A fixed feedforward MLP is used across every simulation.
(Hidden layers 64 -> 32 -> 16, 30% dropout, 5000 epochs, Adam lr 1e-4.) 
LRP is applied with mixed rules: **ε-rule** at the top hidden layer, **γ-rule**in the middle, and a **w²-rule** at the input.

---

## Attribution Results

### 1. Uncorrelated linear

The most important feature (largest true coefficient) is easily identified, with the next-most-important features trailing. 
Ranking matches the ground truth but per-observation magnitudes are noisy.

<br>
<img width="600" alt="LRP attribution: uncorrelated linear, unstandardized features, bias on"
src="plots2/linear_unstandardized_bias.png" />
<br>

Same model, LRP rules, and data, only the inputs are now standardized.

<br>
<img width="600" alt="LRP attribution: same setup but features standardized" src="plots2/linear_standardized_bias.png"
 />
<br>

**Takeaway:** LRP recovers the global ranking, but is brittle to preprocessing — standardization alone collapses the
attribution.

---

### 2. Split domain

`y` is generated by two parameter vectors switched on the row's parity. Different features drive even vs. odd rows. Absolute-value heatmap below:

<br>
<img width="600" alt="LRP attribution: split domain, large-magnitude variant, absolute value"
src="plots2/linear_split_unstandardized_bias_large_magnitude_abs.png" />
<br>

The truly relevant features do light up. But the feature that should only matter on odd rows gets attributed on even rows too.

**Takeaway:** LRP finds *which* features matter, not *when*.

---

### 3. Add correlation

Same split-domain target, now run across each of the 25 correlation structures from the top grid. 

**Rows = W skew** (down = correlation more concentrated on a single feature), 
**Columns = D strength** (right = more diagonal-dominated)

<br>
<img width="900" alt="LRP attribution across the correlation sweep, split-domain target"
src="plots2/correlated_linear_split_unstandardized_bias_indicator_correlation_on_all.png" />
<br>

Note on the parameterization: as `D` grows, off-diagonals shrink toward identity, but the residual `WWᵀ` direction concentrates the principal eigenvector. 
So max eigenvector centrality **rises both downward and rightward**, highest at bottom-right and lowest at top-left.

**Takeaway:** Cleaner attribution when centrality is concentrated (bottom-right) but worst where it's diffuse (top-left). 
Entropy didn't matter while concentration did.

---

## Conclusion
- *Linear:* The dominant feature is correctly singled out in the linear case. But who cares about the linear case? The real world is very correlated.
- *Correlated:* Max eigenvector centrality matters more than entropy. LRP can **not** handle 1) preprocessing or 2) interesting correlations.
- *Future work:* Non-correlation dependence (mutual information, conditional distributions), nonlinear ground
truths, architectures beyond the vanilla MLP.
