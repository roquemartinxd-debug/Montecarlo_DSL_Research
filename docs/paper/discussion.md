\## Discussion



The experimental results show that the proposed Monte Carlo DSL introduces a measurable runtime cost relative to direct vectorized NumPy execution. Across all evaluated models and workload sizes, the native NumPy implementation remained the fastest alternative. This result is expected because the NumPy baseline directly executes optimized numerical operations, whereas the DSL provides an additional abstraction and execution layer.



Therefore, the main contribution of the DSL should not be interpreted as improving the raw numerical performance of NumPy. Instead, the system provides a higher-level environment for specifying, validating, generating, executing, and observing Monte Carlo simulations while preserving reproducible execution behavior.



\### Performance–Abstraction Trade-off



The comparison with `numpy\_native` quantifies the computational cost associated with the DSL infrastructure. For the evaluated workloads, the single-worker DSL required approximately 2.6 times the end-to-end runtime of native NumPy at 10,000 and 100,000 iterations, with the relative difference increasing at 1,000,000 iterations.



This overhead must be interpreted in the context of the functionality provided by the DSL. A model expressed through the DSL is processed through semantic analysis, code generation, deterministic random-number configuration, batch-oriented execution, statistical aggregation, non-finite result handling, percentile computation, and structured runtime reporting.



The benchmark therefore measures the cost of the complete abstraction rather than only the numerical sampling kernel.



An important observation is that DSL compilation itself accounted for only a small fraction of total execution time. Compilation was on the order of a few milliseconds in the evaluated models, indicating that most of the measured overhead occurs during execution of the generated simulation and its associated runtime infrastructure rather than during source translation.



\### Scaling Relative to the Reference Statistical Implementation



The comparison with `python\_reference` reveals a different scaling pattern.



At smaller workloads, the reference implementation was faster than the DSL. However, as the number of iterations increased, the performance gap decreased and eventually reversed at 1,000,000 iterations. At this workload, the single-worker DSL required approximately 30–39% less end-to-end time than the reference implementation, depending on the model.



This result should not be interpreted as evidence that the DSL outperforms optimized Python numerical computing in general. The independent NumPy baseline clearly demonstrates otherwise.



Instead, the result indicates that the generated batch-oriented execution architecture scales more favorably than the specific reference path based on the project's `RunningStats` statistical implementation.



The distinction between the two baselines is therefore important. `numpy\_native` characterizes the performance of an efficient vectorized implementation, while `python\_reference` provides a reference implementation that uses the same statistical infrastructure employed within the project.



\### Parallel Execution



The experiments did not demonstrate a performance advantage from increasing the DSL configuration from one to two workers.



For the smallest workload, the configured batch size of 25,000 samples means that a 10,000-iteration simulation contains only one effective batch, leaving no meaningful opportunity for concurrent batch execution.



At 100,000 and 1,000,000 iterations, the two-worker configuration was consistently slower than the single-worker configuration for the evaluated models. The additional cost associated with process management, task coordination, data transfer, and result aggregation therefore exceeded the computational benefit obtained from parallel execution under the tested conditions.



This behavior suggests that worker count should not be treated as an automatically beneficial configuration parameter. Parallel execution is likely to depend on workload size, batch granularity, model complexity, and the relative cost of inter-process coordination.



Consequently, future optimization should consider adaptive batch sizing and worker selection rather than assuming that additional workers necessarily reduce execution time.



\### Statistical Reliability and Runtime Observability



Performance is only one aspect of the proposed system.



The analytical validation experiments showed that the DSL-generated simulations reproduced known theoretical expectations for the evaluated normal, uniform, discrete, and linear-combination models across multiple seeds. Estimation errors remained within the predefined Monte Carlo uncertainty criterion used in the validation procedure.



The non-finite validation experiment also demonstrated that the runtime can identify invalid numerical results, discard them according to the configured policy, preserve candidate accounting, and report valid and discarded proportions.



These capabilities are particularly relevant for reproducible simulation workflows because numerical failures are not silently incorporated into the reported statistics.



The runtime additionally exposes execution metadata such as random-number generation strategy, statistical configuration, candidate counts, validity rates, and environment information. This information improves traceability compared with an ad hoc simulation script in which equivalent metadata must be implemented manually.



\### Reproducibility as a Design Objective



The DSL explicitly controls the master random seed and derives batch-level random-number streams through a deterministic seed strategy. This makes the relationship between the model specification and the generated execution behavior explicit.



The benchmark methodology follows the same principle. Simulation seeds, scheduling seeds, implementation configurations, repetition counts, software environment information, and the Git commit associated with the benchmark execution are recorded with the raw results.



This level of provenance is relevant for experimental software because it separates reproducibility of the simulation from reproducibility of the performance evaluation itself.



\### Implications



The experimental evidence supports positioning the DSL as a reproducible simulation abstraction rather than as a replacement for direct high-performance NumPy programming.



For users whose primary objective is minimum execution latency for a small number of manually implemented models, direct NumPy remains the more appropriate approach.



The DSL becomes more relevant when the objective includes additional concerns such as declarative model specification, semantic validation, repeatable random-number configuration, standardized statistical outputs, explicit treatment of invalid numerical results, runtime metadata, and generated execution code.



The practical value of the system therefore lies in reducing the amount of infrastructure that must be implemented separately for each Monte Carlo experiment while providing a consistent execution model.



The results also identify clear optimization opportunities. In particular, the performance gap relative to native NumPy indicates that future work should focus on reducing runtime overhead, improving statistical aggregation, and determining batch and worker configurations dynamically according to workload characteristics.



Overall, the experiments characterize a deliberate trade-off: the proposed DSL sacrifices part of the raw performance available through direct vectorized NumPy execution in exchange for a higher-level, reproducible, and observable Monte Carlo simulation workflow.

