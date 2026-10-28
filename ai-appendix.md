# AI Transcript
(https://github.com/jioh-shin/econ5200/REPO/blob/main/ai-appendix.md)
Model: ChatGPT GPT-6
Date: October 27, 2026

## Version 1 Prompt
```css
[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.
```

## Version 1 Reply
## 1. Approach: Mixture Sequential Probability Ratio Test (mSPRT)

I recommend a mixture e-process, an always-valid sequential testing method based on the likelihood-ratio principle.

Unlike a conventional fixed-sample t-test, it permits checking results after every new observation without exceeding the specified Type I error probability, provided the test assumptions hold.

Target Type I error

# 5%

Null simulations

# 1,000

Maximum looks

# 2,000

### Test assumptions

I'll use a binary-outcome A/B experiment, with independent Bernoulli outcomes and one treatment-control pair per look.

The log format is:

```
{    "experiment_id": "AB_001",    "control": 0,    "treatment": 1}
```

Your actual log schema wasn't supplied, so the input-field mapping is isolated in the implementation.

Define the difference for each observation pair:

\\[ D_i=Y\_{T,i}-Y\_{C,i} \\]

Under the null hypothesis:

\\[ H_0:p_T=p_C \\]

We construct a nonnegative e-process \\(E_n\\) satisfying:

\\[ P\_{H_0}\left(\sup\_{n\geq1}E_n\geq\frac1\alpha\right)\leq\alpha \\]

At \\(\alpha=0.05\\), the stopping rule is:

\\[ \boxed{\text{Reject }H_0\text{ when }E_n\geq20} \\]

This controls the probability of ever falsely rejecting the null, not just the probability at one predetermined sample size.

## 2. Python implementation

This implementation accepts a stream of observations and yields a stopping decision after each look. It uses log-space calculations for numerical stability.

```

import numpy as np

def sequential_ab_test(
    stream,
    alpha=0.05,
    tau=0.5
):
    """
    Anytime-valid two-sided A/B test for independent
    Bernoulli outcomes, observed in treatment-control pairs.

    Yields one decision for every incoming observation pair.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if tau <= 0:
        raise ValueError("tau must be positive")

    n = 0
    cumulative_diff = 0.0
    max_log_e = 0.0
    stopped = False

    for row in stream:
        control = row["control"]
        treatment = row["treatment"]

        if control not in (0, 1) or treatment not in (0, 1):
            raise ValueError("Binary outcomes must be 0 or 1")

        n += 1
        cumulative_diff += treatment - control

        # Bernoulli-difference sub-Gaussian e-process
        v = 1.0 + n * tau**2 / 2.0

        log_e = (
            -0.5 * np.log(v)
            + tau**2 * cumulative_diff**2 / (2.0 * v)
        )

        max_log_e = max(max_log_e, log_e)

        # Anytime-valid p-value
        p_anytime = float(np.exp(-max_log_e))

        if log_e >= np.log(1.0 / alpha):
            stopped = True

        yield {
            "experiment_id": row.get("experiment_id"),
            "look": n,
            "effect_estimate": cumulative_diff / n,
            "e_value": float(np.exp(log_e)),
            "p_anytime": p_anytime,
            "reject_null": stopped,
            "stop": stopped
        }

```

Here, `tau` determines the mixture's sensitivity to different effect sizes. It must be selected before examining the experiment's results.

The returned `stop` decision is persistent: once the test rejects, subsequent records remain marked as stopped. In production, data collection can terminate at that point.

## 3. Evaluation: 1,000 pure-null streams

I ran a reproducible simulation with the following parameters:

- 1,000 independent A/B experiments, all with true effect zero.
- Control conversion probability = treatment conversion probability = 0.50.
- 2,000 treatment-control pairs per experiment.
- Significance level \\(\alpha=0.05\\).
- Monitoring after every pair.

For comparison, I also evaluated a naive two-sided z-test with a 5% threshold at every look. It uses the correct null variance for these Bernoulli differences.

Observed false-positive rates

0%20%40%60%80%Naive repeated testingAlways-valid mSPRT

1,000 null experiments; 2,000 looks per experiment

| Method                    | False rejections | Rate  |
| ------------------------- | ---------------- | ----- |
| Naive repeated z-test     | 578 / 1,000      | 57.8% |
| Always-valid mixture test | 33 / 1,000       | 3.3%  |
| Nominal upper bound       | 50 / 1,000       | 5.0%  |

The sequential method controlled false positives in this experiment.

### Reproduce the evaluation

```

import numpy as np
from scipy.stats import norm

def evaluate_null(
    n_streams=1000,
    max_looks=2000,
    alpha=0.05,
    tau=0.5,
    seed=2026
):
    rng = np.random.default_rng(seed)

    cumulative_diff = np.zeros(n_streams)
    sequential_reject = np.zeros(n_streams, dtype=bool)
    naive_reject = np.zeros(n_streams, dtype=bool)

    critical_z = norm.ppf(1 - alpha / 2)

    for n in range(1, max_looks + 1):

        treatment = rng.binomial(1, 0.5, n_streams)
        control = rng.binomial(1, 0.5, n_streams)
        cumulative_diff += treatment - control

        # Sequential mixture e-process
        v = 1 + n * tau**2 / 2

        log_e = (
            -0.5 * np.log(v)
            + tau**2 * cumulative_diff**2 / (2 * v)
        )

        sequential_reject |= log_e >= np.log(1 / alpha)

        # Naive, repeatedly applied fixed-look z-test
        z = cumulative_diff / np.sqrt(n * 0.5)
        naive_reject |= np.abs(z) > critical_z

    return {
        "sequential_false_positives": int(sequential_reject.sum()),
        "sequential_fpr": float(sequential_reject.mean()),
        "naive_false_positives": int(naive_reject.sum()),
        "naive_fpr": float(naive_reject.mean())
    }

print(evaluate_null())

```

The array-based simulation uses the same e-process formula as the streaming function, allowing efficient validation.

## 4. Mechanism check: Why naive peeking fails

A conventional hypothesis test is calibrated for a fixed sample size.

For example:

\\[ P\_{H_0}(p_n<0.05)\leq0.05 \\]

This does not imply:

\\[ P\_{H_0}\left(\min\_{1\leq n\leq N}p_n<0.05\right)\leq0.05 \\]

When a researcher tests repeatedly, they have multiple chances to cross the significance threshold, even though the null is true.

These tests are correlated, so their joint false-positive probability is not simply \\(1-0.95^N\\). Nevertheless, it typically increases with repeated monitoring.

### What does the sequential construction give up?

| Property                     | Conventional fixed-sample test       | Always-valid mixture test                                 |
| ---------------------------- | ------------------------------------ | --------------------------------------------------------- |
| Type I error                 | Controlled at fixed sample size      | Controlled across all stopping times                      |
| Continuous monitoring        | Not valid without adjustment         | Valid                                                     |
| Critical threshold           | Fixed-look p-value below 0.05        | e-value at least 20                                       |
| Power at a fixed sample size | Often higher when optimally designed | Can be lower                                              |
| Sample-size efficiency       | Efficient for a prespecified test    | May require more observations                             |
| Effect-size assumptions      | Depends on selected test             | Requires a valid outcome model and a prespecified mixture |

The key trade-off is statistical efficiency in exchange for optional-stopping validity.

The sequential test must protect against a false rejection over an entire monitoring path. Therefore, it can require more evidence than a fixed-horizon test designed for one particular sample size.

Our construction is also conservative because it uses a sub-Gaussian bound rather than the exact Bernoulli likelihood.

## 5. Why was the observed rate 3.3% rather than 5%?

This is not evidence that the implementation is incorrect.

The anytime-valid guarantee is:

\\[ \boxed{P\_{H_0}(\text{ever reject})\leq0.05} \\]

It does not require the actual rejection probability to equal 5%.

There are three reasons it may be below 5%:

1. Conservative boundary: The e-process uses an upper bound on the moment-generating function.
2. Finite monitoring horizon: We stopped monitoring after 2,000 looks. Some null paths might cross the threshold later.
3. Monte Carlo uncertainty: With 1,000 streams, the standard error of a measured 5% rate is approximately 0.69 percentage points.

The observed 3.3% is lower than the nominal level, but valid. It indicates conservative finite-horizon behavior rather than inflated Type I error.

A further consideration for your portfolio of 200 experiments: controlling the Type I error at 5% per experiment does not control the portfolio-level family-wise error rate at 5%. If all 200 null hypotheses were true, one could still observe multiple false rejections. Portfolio-wide control requires a separate multiplicity procedure, such as Bonferroni allocation or an appropriate sequential false-discovery-rate method.

## 6. Production validation

Before deployment, I would additionally check:

- Calibration under different baseline conversion rates, including 1%, 10%, and 50%.
- Statistical power under known nonzero treatment effects.
- Detection delay measured in number of observations.
- Correct behavior when a stream never crosses the stopping boundary.
- Independence assumptions, missing observations, and changes in assignment policy.
- Whether experiment-level or portfolio-level error control is required.

Conclusion: The implemented test provides continuous monitoring with a theoretically controlled per-experiment Type I error rate. In the 1,000-stream null evaluation, the empirical false-positive rate was 3.3%, compared with 57.8% for naive repeated testing.

## Version 2 Prompt
[Revision — Version 2]

Your first implementation assumes individual Bernoulli treatment-control observations, but my actual experiment portfolio contains 200 experiments with the columns `true`, `n`, `se`, and `est`.

This is a mismatch with my existing log structure. The portfolio contains final experiment-level summaries, not sequential observations.

Please revise your implementation to:

1. Use a sequential Gaussian observation model consistent with the effect estimates and standard errors in my portfolio.
2. Explain how the sequential observations are generated and why final summary statistics alone are insufficient for reconstructing an actual monitoring history.
3. Implement a statistically valid always-valid stopping rule with Type I error controlled at 5%.
4. Run the revised method on 1,000 pure-null streams and report the empirical false positive rate and Monte Carlo standard error.
5. Compare the revised results with the original implementation and explain any differences.

Use NumPy and SciPy only for the sequential testing implementation. Keep the code compatible with my existing experiment portfolio where possible.

Explain the stopping rule and the assumptions required for its validity.

## Version 2 Reply
# Revised implementation: Gaussian always-valid sequential A/B testing

The revised method uses a Gaussian mixture Sequential Probability Ratio Test (mSPRT), compatible with your portfolio's four columns:

| Column | Interpretation                              | Role                               |
| ------ | ------------------------------------------- | ---------------------------------- |
| `true` | True treatment effect                       | Simulation and power evaluation    |
| `n`    | Final sample size                           | Number of sequential observations  |
| `se`   | Standard error of the final effect estimate | Calibrates observation variance    |
| `est`  | Observed final effect estimate              | Used for comparison and validation |

The crucial distinction is that your 200 rows describe 200 final experiment outcomes, not the intermediate observations required to reconstruct their monitoring histories.

## 1. Statistical model

For experiment \\(j\\), assume the sequential effect observations follow

\\[ X\_{j,t}\overset{iid}{\sim}N(\theta_j,\sigma_j^2) \\]

where \\(\theta_j\\) is the true treatment effect.

The final effect estimate and standard error are

\\[ \hat\theta_j=\frac1{n_j}\sum\_{t=1}^{n_j}X\_{j,t}, \qquad SE_j=\frac{\sigma_j}{\sqrt{n_j}}. \\]

Therefore, under this model, the observation-level standard deviation is

\\[ \boxed{\sigma_j=SE_j\sqrt{n_j}}. \\]

This allows us to generate sequential observations from each experiment's final `n`, `se`, and `true`.

For example, a portfolio row with `true = 0`, `n = 1000`, and `se = 0.04` implies an observation standard deviation of approximately 1.265. We can simulate 1,000 normally distributed observations with that variance.

Important qualification: This relationship is exact only under the assumed Gaussian sampling model with known variance. If your reported `se` is estimated, heteroskedasticity-robust, cluster-robust, or otherwise adjusted, the same conversion does not automatically justify an exact Gaussian mSPRT.

### Why the final summaries are insufficient

The final `est` records only the average at the end of the experiment. It does not contain the sequence of partial estimates.

Two experiments can have identical final `est`, `n`, and `se` but different interim paths. One may cross the sequential rejection boundary early, while the other never crosses it.

Consequently:

- We can simulate plausible sequential histories using the portfolio parameters.
- We cannot recover the experiments' actual historical stopping decisions without their intermediate observations or sufficient-statistic logs.
- Ordinary simulated paths do not need to reproduce the observed `est` exactly; they reproduce its assumed sampling distribution. Conditional simulation could force the final estimate to match, but would still not identify the original path.

## 2. Always-valid stopping rule

We test

\\[ H_0:\theta=0 \qquad\text{against}\qquad H_1:\theta\ne0. \\]

Choose a Gaussian mixture under the alternative:

\\[ \theta\sim N(0,\tau^2). \\]

Let \\(S_t=\sum\_{i=1}^{t}X_i\\) and \\(I_t=t/\sigma^2\\).

The resulting mixture likelihood ratio is

\\[ E_t= \frac{1}{\sqrt{1+\tau^2I_t}} \exp\left( \frac{\tau^2(S_t/\sigma^2)^2} {2(1+\tau^2I_t)} \right). \\]

Under the stated Gaussian null, \\(E_t\\) is a nonnegative martingale with initial expectation one. Ville's inequality then gives

\\[ P\_{H_0}\left(\sup\_{t\ge1}E_t\ge20\right)\le0.05. \\]

The rule is therefore:

\\[ \boxed{\text{Stop and reject when } E_t\ge20.} \\]

No prespecified final look is necessary for Type I error control. However, the variance model and mixture parameter must be fixed independently of the accumulating results.

## 3. Evaluation results

I executed a reproducible simulation of 1,000 pure-null Gaussian streams, alongside a fresh run of the original Bernoulli construction.

Each stream was monitored at every look through 2,000 observations or observation pairs, with \\(\alpha=0.05\\), mixture scale \\(\tau=0.5\\), and random seed `20261027`.

Revised Gaussian mSPRT

# 3.0%

30 false positives / 1,000

Monte Carlo SE

# 0.54 pp

Estimated simulation uncertainty

0%15%30%45%60%Gaussian mSPRTBernoulli mSPRTGaussian naiveBernoulli naive

Empirical probability of at least one false rejection across 2,000 looks; 1,000 streams per method.

| Method                          | False positives | Empirical FPR | MC SE   |
| ------------------------------- | --------------- | ------------- | ------- |
| Revised Gaussian mSPRT          | 30              | 3.0%          | 0.54 pp |
| Original Bernoulli e-process    | 34              | 3.4%          | 0.57 pp |
| Naive Gaussian repeated z-test  | 561             | 56.1%         | 1.57 pp |
| Naive Bernoulli repeated z-test | 568             | 56.8%         | 1.57 pp |

The new Gaussian test passes the null calibration check: its observed false-positive rate is below the 5% bound. A rate of exactly 5% is not required.

These are independent simulations of 1,000 streams, not a backtest on your 200 portfolio records.

## 4. Implementation: NumPy and SciPy only

The following code supports streaming observations, retains the existing portfolio structure, and reproduces the Gaussian null simulation.

```

import numpy as np
from scipy.stats import norm

def gaussian_msprt(stream, sigma, alpha=0.05, tau=0.5):
    """
    Two-sided, anytime-valid Gaussian mixture SPRT.

    stream: iterable of observations or dictionaries
            with an 'x' observation field
    sigma: known observation-level standard deviation
    alpha: Type I error upper bound
    tau: prespecified alternative-prior standard deviation

    Yields a decision at each look.
    """
    if sigma <= 0 or tau <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid sigma, tau, or alpha")

    n = 0
    total = 0.0
    max_log_e = 0.0
    rejected = False

    for row in stream:
        x = float(row["x"] if isinstance(row, dict) else row)

        if not np.isfinite(x):
            raise ValueError("Observations must be finite")

        n += 1
        total += x

        information = n / sigma**2
        score = total / sigma**2
        v = 1.0 + tau**2 * information

        log_e = -0.5 * np.log(v) + (
            tau**2 * score**2 / (2.0 * v)
        )

        max_log_e = max(max_log_e, log_e)
        rejected |= log_e >= np.log(1.0 / alpha)

        yield {
            "look": n,
            "est": total / n,
            "se": sigma / np.sqrt(n),
            "log_e": float(log_e),
            "p_anytime": float(np.exp(-max_log_e)),
            "stop": bool(rejected),
            "reject_null": bool(rejected)
        }


def simulate_portfolio_experiment(row, seed=2026,
                                  alpha=0.05, tau=0.5):
    """
    Simulate a Gaussian sequential path from one
    existing portfolio row: true, n, se, est.
    """
    true_effect = float(row["true"])
    n_final = int(row["n"])
    final_se = float(row["se"])

    if n_final < 1 or final_se <= 0:
        raise ValueError("Invalid n or se")

    sigma = final_se * np.sqrt(n_final)
    rng = np.random.default_rng(seed)

    # Generate a new path using the portfolio's parameters.
    observations = rng.normal(
        loc=true_effect,
        scale=sigma,
        size=n_final
    )

    history = list(
        gaussian_msprt(observations, sigma, alpha, tau)
    )

    # First look at which rejection occurred, if any.
    stopping_look = next(
        (d["look"] for d in history if d["stop"]),
        None
    )

    return {
        "true": true_effect,
        "n": n_final,
        "se": final_se,
        "est": float(row["est"]),
        "simulated_final_est": float(observations.mean()),
        "stopping_look": stopping_look,
        "rejected": stopping_look is not None,
        "history": history
    }

```

For an existing portfolio represented as a list of dictionaries, use:

```
results = [    simulate_portfolio_experiment(row, seed=2026 + i)    for i, row in enumerate(portfolio)]for r in results:    print(        r["true"],        r["est"],        r["stopping_look"],        r["rejected"]    )
```

This is directly compatible with the field names `true`, `n`, `se`, and `est`. A pandas DataFrame can also be converted with `portfolio.to_dict("records")`, without requiring pandas in the testing implementation.

### Reproduce the pure-null evaluation

````
   
def evaluate_gaussian_null(
    n_streams=1000,
    max_looks=2000,
    alpha=0.05,
    sigma=1.0,
    tau=0.5,
    seed=20261027
):
    rng = np.random.default_rng(seed)

    sums = np.zeros(n_streams)
    rejected = np.zeros(n_streams, dtype=bool)
    naive_rejected = np.zeros(n_streams, dtype=bool)

    threshold = np.log(1 / alpha)
    critical_z = norm.ppf(1 - alpha / 2)

    for n in range(1, max_looks + 1):
        observations = rng.normal(0, sigma, n_streams)
        sums += observations

        information = n / sigma**2
        score = sums / sigma**2
        v = 1 + tau**2 * information

        log_e = (
            -0.5 * np.log(v)
            + tau**2 * score**2 / (2 * v)
        )

        rejected |= log_e >= threshold

        # Fixed-look z-test applied repeatedly
        z = sums / (sigma * np.sqrt(n))
        naive_rejected |= np.abs(z) > critical_z

    fpr = rejected.mean()
    mc_se = np.sqrt(fpr * (1 - fpr) / n_streams)

    return {
        "false_positives": int(rejected.sum()),
        "empirical_fpr": float(fpr),
        "monte_carlo_se": float(mc_se),
        "naive_fpr": float(naive_rejected.mean())
    }

print(evaluate_gaussian_null())
```

The execution produced:

```python
{
    'false_positives': 30,
    'empirical_fpr': 0.03,
    'monte_carlo_se': 0.005394441583704471,
    'naive_fpr': 0.561
}
```

## 5. Comparison with Version 1

The Bernoulli construction used paired treatment-control differences, which are discrete and bounded. The revised Gaussian construction uses continuous measurements with known observation variance.

Both support continuous monitoring, but their likelihood assumptions and the e-process calibration differ.

The earlier response reported a 3.3% false-positive rate for the Bernoulli implementation. In this new reproducible evaluation, the Bernoulli rate was 3.4%, while the Gaussian rate was 3.0%. That 0.4-percentage-point difference is small relative to the Monte Carlo uncertainty and does not establish that one method provides better Type I error control.

The important change is **model compatibility**, not the small change in observed false-positive rate.

## 6. Conditions required for validity

The exact guarantee for this implementation requires independent Gaussian observations with a constant, correctly specified, known variance under the null. It also requires the alternative mixture scale `tau` to be fixed before monitoring, and the e-process to be updated using all observations in their actual order.

If experiments have unequal observation variances, correlated outcomes, estimated standard errors, or adaptive enrollment, the implementation may require modification. A valid Gaussian e-process can accommodate predictable observation-specific variances, but blindly using a final robust standard error is not sufficient.

One final distinction is necessary: this controls the probability of a false rejection at 5% **for each experiment**. It does not control the probability of any false rejection across the entire portfolio of 200 experiments. If portfolio-wide family-wise error control is required, allocate significance levels across experiments, for example with Bonferroni's \(\alpha_j=0.05/200\).

**Final assessment:** The revised Gaussian mSPRT is statistically valid under its specified sampling model, accepts your portfolio's existing summary fields for simulation, and achieved an empirical null false-positive rate of **3.0% ± 0.54 percentage points (one Monte Carlo standard error)** over 1,000 streams. Actual historical stopping behavior remains unidentified from the four final summary columns alone.
````
## Changes made
In Version 2, I changed the Bernoulli model to a Gaussian model because my portfolio only contains summary data. Since the original data doesn't show how the results changed over time, I asked AI to explain this limitation and simulate sequential observations instead. I also asked it to test the method on 1,000 null streams, calculate the false positive rate and Monte Carlo SE, and compare the results with Version 1.
