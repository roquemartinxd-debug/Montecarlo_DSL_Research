\## Threats to Validity



The experimental evaluation was designed to provide reproducible evidence about the statistical correctness and runtime behavior of the proposed Monte Carlo DSL. Nevertheless, several limitations should be considered when interpreting the results.



\### Internal Validity



Execution-time measurements may be affected by operating-system scheduling, background processes, memory management, process startup costs, and other sources of runtime variability. To reduce these effects, each benchmark configuration was executed 30 times after two warm-up runs, and the execution order was randomized using a deterministic scheduling seed.



All evaluated implementations were executed in fresh Python subprocesses. Consequently, the reported measurements represent end-to-end latency rather than isolated numerical-kernel performance. For the DSL configurations, total execution time additionally includes DSL compilation and generated-script creation.



Timing outliers were retained rather than removed. Because several configurations exhibited right-tailed observations, median execution times and paired runtime ratios were used as the primary performance measures, together with bootstrap confidence intervals.



\### Construct Validity



The native NumPy baseline and the Python reference implementation represent different computational strategies.



The `numpy\_native` implementation uses direct vectorized NumPy operations and serves as the primary reference for numerical performance. In contrast, the `python\_reference` implementation uses the project's `RunningStats` statistical infrastructure.



Therefore, comparisons against the Python reference implementation should not be interpreted as comparisons against optimized NumPy. The two baselines were intentionally retained to distinguish the cost of the DSL from the behavior of the project's reference statistical implementation.



The reported DSL timings measure the complete execution pipeline, including compilation and generated-program execution. Consequently, the benchmark evaluates the practical runtime cost of using the DSL rather than only the cost of random-number generation or numerical computation.



\### External Validity



The performance experiments considered four representative stochastic models: normal, uniform, discrete, and linear-combination models. Although these cases cover different probability distributions and model structures, they do not represent the complete range of Monte Carlo applications.



The evaluated workloads were limited to 10,000, 100,000, and 1,000,000 iterations. Different model complexity, larger simulations, alternative batch sizes, or workloads involving more expensive mathematical expressions may exhibit different scaling behavior.



The experiments were also conducted on a single software and hardware environment. Consequently, the reported execution times should not be interpreted as universal performance characteristics across operating systems, processors, Python versions, or NumPy versions.



\### Parallelism



The evaluation considered one-worker and two-worker DSL configurations using a fixed batch size of 25,000 samples.



At 10,000 iterations, the workload contains fewer samples than a single configured batch and therefore provides no meaningful opportunity for parallel batch execution. At larger workloads, the two-worker configuration did not outperform the single-worker configuration under the evaluated environment.



These results should therefore be interpreted only for the tested worker counts, batch size, models, and execution platform. They do not establish that parallel execution is generally ineffective for larger workloads or different computational models.



\### Statistical Generalization



Analytical validation demonstrated agreement with known theoretical results for the selected validation models, and the controlled non-finite experiment verified the implemented discard-and-report behavior for a specific partially invalid model.



These experiments provide evidence for the tested statistical mechanisms but do not constitute a formal proof of correctness for every model expressible in the DSL.



Additional distributions, dependent-variable structures, extreme numerical conditions, and domain-specific simulations remain relevant directions for broader validation.

