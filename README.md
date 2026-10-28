# The Underpowered Portfolio: A/B Testing and Decision Rules

## Objective
I examined how statistical significance, low statistical power, and different decision rules affect the reliability of A/B testing results.

## Methodology
- Simulated a portfolio of 200 A/B experiments, assuming 80% had no true effect and 20% had positive effects.
- Applied a p-value threshold of 0.05 to identify significant experiments and measured the share of false positives.
- Examined the winner's curse by comparing estimated and true effects across different power groups.
- Used bootstrap methods to calculate percentile and BCa confidence intervals for the total true effect.
- Compared the original p < 0.05 rule with a stricter p < 0.01 rule.
- Used AI to explore sequential testing and compared the false positive rates of Bernoulli and Gaussian models.

## Key Findings
- Out of 200 experiments, 49 were selected under the p < 0.05 rule, and 24.5% of the selected experiments had no true effect.
- The overall winner's curse inflation factor was 1.022, with the greatest exaggeration in the low-power group (1.104).
- The total true effect of selected experiments was 11.365, with a 95% bootstrap percentile confidence interval of [8.031, 14.886].
- Using the stricter p < 0.01 rule reduced the number of selected experiments from 49 to 38 and lowered the false-ship rate from 24.5% to 2.6%, while keeping the same total realised return of 11.365.
- In 1,000 pure-null simulations, the Gaussian sequential test had a false positive rate of 3.0%, compared with 56.1% for naive repeated testing.

Overall, I found that relying only on statistical significance can lead to false positives and exaggerated estimates. Stricter decision rules and sequential testing can help improve the reliability of experiment decisions, although they involve trade-offs in statistical power and sample size.
