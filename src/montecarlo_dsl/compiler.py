from __future__ import annotations

from dataclasses import dataclass

from .generator import GeneratorConfig, PythonCodeGenerator
from .semantic import ValidatedProgram, analyze_source


@dataclass(frozen=True, slots=True)
class CompiledSimulation:
    """Resultado inmutable de la fase de compilacion previa a la ejecucion."""

    program: ValidatedProgram
    script: str
    seed: int

    @property
    def iterations(self) -> int:
        return self.program.iterations


def compile_program(
    program: ValidatedProgram,
    config: GeneratorConfig | None = None,
) -> CompiledSimulation:
    """Genera el script ejecutable sin crear ni iniciar ningun trabajo runtime."""

    generator = PythonCodeGenerator(program, config)
    script = generator.generate()
    return CompiledSimulation(program=program, script=script, seed=generator.seed)


def compile_source(
    source: str,
    config: GeneratorConfig | None = None,
) -> CompiledSimulation:
    """Ejecuta LEX/SYN/SEM/GEN de forma completa antes de la fase runtime."""

    return compile_program(analyze_source(source), config)
