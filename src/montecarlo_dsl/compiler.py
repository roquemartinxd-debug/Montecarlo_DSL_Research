from __future__ import annotations

from dataclasses import dataclass

from .cost_model import ASTCostFeatures, InterpretableASTCostModel
from .generator import GeneratorConfig, PythonCodeGenerator
from .optimizer import OptimizationMetrics
from .scheduler import ExecutionSchedulePlan
from .semantic import ValidatedProgram, analyze_source


@dataclass(frozen=True, slots=True)
class CompiledSimulation:
    """Resultado inmutable de la fase de compilacion previa a la ejecucion."""

    program: ValidatedProgram
    script: str
    seed: int
    optimization_metrics: OptimizationMetrics | None = None
    cost_features: ASTCostFeatures | None = None
    static_schedule_plan: ExecutionSchedulePlan | None = None
    cost_model: InterpretableASTCostModel | None = None

    @property
    def iterations(self) -> int:
        return self.program.iterations


def compile_program(
    program: ValidatedProgram,
    config: GeneratorConfig | None = None,
    cost_model: InterpretableASTCostModel | None = None,
) -> CompiledSimulation:
    """Ejecuta optimizacion AST, estimacion de coste, planificacion estatica y generacion de codigo."""

    effective_config = config or GeneratorConfig()
    effective_cost_model = cost_model or InterpretableASTCostModel()
    generator = PythonCodeGenerator(
        program,
        effective_config,
        cost_model=effective_cost_model,
    )
    script = generator.generate()
    return CompiledSimulation(
        program=generator.program,
        script=script,
        seed=generator.seed,
        optimization_metrics=generator.optimization_metrics,
        cost_features=generator.cost_features,
        static_schedule_plan=generator.static_schedule_plan,
        cost_model=effective_cost_model,
    )


def compile_source(
    source: str,
    config: GeneratorConfig | None = None,
    cost_model: InterpretableASTCostModel | None = None,
) -> CompiledSimulation:
    """Ejecuta LEX/SYN/SEM/OPT/COST/GEN de forma completa antes de la fase runtime."""

    max_it = config.max_iterations if config is not None else 1_000_000
    return compile_program(
        analyze_source(source, max_iterations=max_it),
        config,
        cost_model=cost_model,
    )
