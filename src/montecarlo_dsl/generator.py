from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import secrets
from textwrap import dedent, indent

from .ast import (
    BinaryOperationNode,
    BinaryOperator,
    DiscreteDistributionNode,
    ExpressionNode,
    FractionNode,
    NormalDistributionNode,
    NumberLiteralNode,
    PlusMinusNode,
    PowerNode,
    SqrtNode,
    UnaryOperationNode,
    UnaryOperator,
    UniformDistributionNode,
    VariableReferenceNode,
)
from .diagnostics import CompilationError, Diagnostic, DiagnosticPhase
from .semantic import ValidatedProgram, analyze_source
from .source import SourceSpan


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Configuracion del script Monte Carlo generado."""

    batch_size: int = 25_000
    max_workers: int = 4
    histogram_bins: int = 30
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be greater than zero")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ExpressionCode:
    """Variantes vectorizadas producidas a partir de una expresion."""

    branches: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("ExpressionCode requires at least one branch")
        if len(self.branches) > 2:
            raise ValueError("The validated DSL can produce at most two branches")


class NumpyExpressionVisitor:
    """Traduce el AST validado a expresiones NumPy vectorizadas."""

    def __init__(self, variable_names: dict[str, str] | None = None):
        self.variable_names = variable_names or {}

    def visit(self, node: ExpressionNode) -> ExpressionCode:
        if isinstance(node, NumberLiteralNode):
            return ExpressionCode((self._decimal(node.value, node.span),))

        if isinstance(node, VariableReferenceNode):
            return ExpressionCode((self.variable_names.get(node.name, node.name),))

        if isinstance(node, UnaryOperationNode):
            operand = self.visit(node.operand)
            symbol = "+" if node.operator is UnaryOperator.PLUS else "-"
            return ExpressionCode(tuple(f"({symbol}({branch}))" for branch in operand.branches))

        if isinstance(node, BinaryOperationNode):
            left = self.visit(node.left)
            right = self.visit(node.right)
            symbol = {
                BinaryOperator.ADD: "+",
                BinaryOperator.SUBTRACT: "-",
                BinaryOperator.MULTIPLY: "*",
                BinaryOperator.DIVIDE: "/",
            }[node.operator]
            return self._combine(left, right, lambda a, b: f"(({a}) {symbol} ({b}))")

        if isinstance(node, PlusMinusNode):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if len(left.branches) != 1 or len(right.branches) != 1:
                # Mantiene el contrato aunque el programa validado se construya manualmente.
                self._error(
                    "GEN002",
                    "El AST contiene ramas incompatibles con la restriccion de un solo \\pm.",
                    node.span,
                )
            a = left.branches[0]
            b = right.branches[0]
            return ExpressionCode((f"(({a}) + ({b}))", f"(({a}) - ({b}))"))

        if isinstance(node, PowerNode):
            base = self.visit(node.base)
            exponent = self.visit(node.exponent)
            return self._combine(
                base,
                exponent,
                lambda a, b: f"np.power(({a}), ({b}))",
            )

        if isinstance(node, FractionNode):
            numerator = self.visit(node.numerator)
            denominator = self.visit(node.denominator)
            return self._combine(
                numerator,
                denominator,
                lambda a, b: f"(({a}) / ({b}))",
            )

        if isinstance(node, SqrtNode):
            expression = self.visit(node.expression)
            return ExpressionCode(tuple(f"np.sqrt(({branch}))" for branch in expression.branches))

        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    @staticmethod
    def _combine(
        left: ExpressionCode,
        right: ExpressionCode,
        render,
    ) -> ExpressionCode:
        branches = tuple(render(a, b) for a in left.branches for b in right.branches)
        if len(branches) > 2:
            raise ValueError("More than two expression branches were produced")
        return ExpressionCode(branches)

    @staticmethod
    def _float64(value: Decimal, span: SourceSpan) -> float:
        converted = float(value)
        if not math.isfinite(converted):
            raise CompilationError(
                Diagnostic(
                    code="GEN001",
                    phase=DiagnosticPhase.GENERATION,
                    message="Un literal numerico no puede representarse como float64 finito para NumPy.",
                    span=span,
                    lexeme=str(value),
                )
            )
        if value != 0 and converted == 0.0:
            raise CompilationError(
                Diagnostic(
                    code="GEN004",
                    phase=DiagnosticPhase.GENERATION,
                    message=(
                        "Un literal numerico no nulo se pierde por underflow al "
                        "convertirse a float64 para NumPy."
                    ),
                    span=span,
                    lexeme=str(value),
                )
            )
        return converted

    @staticmethod
    def _decimal(value: Decimal, span: SourceSpan) -> str:
        converted = NumpyExpressionVisitor._float64(value, span)
        return repr(converted)

    @staticmethod
    def _error(code: str, message: str, span: SourceSpan) -> None:
        raise CompilationError(
            Diagnostic(
                code=code,
                phase=DiagnosticPhase.GENERATION,
                message=message,
                span=span,
            )
        )


class PythonCodeGenerator:
    """Genera un script Python/NumPy autocontenido desde ``ValidatedProgram``."""

    def __init__(self, program: ValidatedProgram, config: GeneratorConfig | None = None):
        self.program = program
        self.config = config or GeneratorConfig()
        self.seed = self.config.seed if self.config.seed is not None else secrets.randbits(128)
        self.variable_names = {
            name: f"_mc_var_{index}"
            for index, name in enumerate(self.program.random_variables)
        }
        self.expression_visitor = NumpyExpressionVisitor(self.variable_names)

    def generate(self) -> str:
        expression = self.expression_visitor.visit(self.program.ast.model.expression)
        expected_branches = 2 if self.program.has_plus_minus else 1
        if len(expression.branches) != expected_branches:
            self._error(
                "GEN003",
                "La cantidad de ramas generadas no coincide con el resultado del analisis semantico.",
                self.program.ast.model.expression.span,
            )

        distribution_lines = self._distribution_lines()
        result_lines = self._result_lines(expression)
        requested = ", ".join(repr(item.value) for item in self.program.requested_statistics)
        branch_ids = ("plus", "minus") if len(expression.branches) == 2 else ("main",)
        branch_labels = ("Rama +", "Rama -") if len(expression.branches) == 2 else ("Resultado",)

        template = f'''\
# Generado por montecarlo_dsl.
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import sys
import traceback

import numpy as np

PROTOCOL_VERSION = 2
EVENT_PREFIX = "MC_DSL_EVENT:"
MASTER_SEED = {self.seed}
TOTAL_ITERATIONS = {self.program.iterations}
BATCH_SIZE = {self.config.batch_size}
MAX_WORKERS = {self.config.max_workers}
HISTOGRAM_BINS = {self.config.histogram_bins}
REQUESTED_STATISTICS = ({requested},)
BRANCH_IDS = {branch_ids!r}
BRANCH_LABELS = {branch_labels!r}
BRANCH_COUNT = len(BRANCH_IDS)


def _emit(payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(EVENT_PREFIX + encoded, flush=True)


def _batch_seed(batch_index):
    material = f"{{MASTER_SEED}}:{{batch_index}}".encode("ascii")
    digest = hashlib.blake2b(material, digest_size=16).digest()
    return int.from_bytes(digest, "big")


def _stats_payload(average, minimum, maximum):
    available = {{"avg": average, "min": minimum, "max": maximum}}
    return {{name: available[name] for name in REQUESTED_STATISTICS}}


def _numerical_diagnostics(valid_results, total_sum, total_sum_sq):
    if valid_results <= 0:
        return {{
            "n": 0,
            "sample_stddev": None,
            "mcse_mean": None,
            "ci95_mean": None,
            "relative_mcse": None,
        }}

    mean = total_sum / valid_results
    if valid_results > 1:
        numerator = total_sum_sq - (total_sum * total_sum) / valid_results
        variance = max(0.0, numerator / (valid_results - 1))
        sample_stddev = math.sqrt(variance)
        mcse = sample_stddev / math.sqrt(valid_results)
        ci95 = [mean - 1.96 * mcse, mean + 1.96 * mcse]
        relative = abs(mcse / mean) if mean != 0 else None
    else:
        sample_stddev = None
        mcse = None
        ci95 = None
        relative = None

    return {{
        "n": int(valid_results),
        "sample_stddev": sample_stddev,
        "mcse_mean": mcse,
        "ci95_mean": ci95,
        "relative_mcse": relative,
    }}


def _histogram_payload(counts, edges, data_min, data_max):
    if counts is None or edges is None:
        return {{"data_min": None, "data_max": None, "total_count": 0, "bins": []}}
    bins = [
        {{"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}}
        for i in range(len(counts))
    ]
    return {{
        "data_min": float(data_min),
        "data_max": float(data_max),
        "total_count": int(np.sum(counts, dtype=np.int64)),
        "bins": bins,
    }}


def _histogram_edges(data_min, data_max):
    if data_min == data_max:
        padding = max(abs(data_min) * 0.01, 0.5)
        low = data_min - padding
        high = data_max + padding
    else:
        low = data_min
        high = data_max
    return np.linspace(low, high, HISTOGRAM_BINS + 1, dtype=np.float64)


def _recompute_histogram(chunks, data_min, data_max):
    edges = _histogram_edges(data_min, data_max)
    counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    for chunk in chunks:
        chunk_counts, _ = np.histogram(chunk, bins=edges)
        counts += chunk_counts.astype(np.int64, copy=False)
    return counts, edges


def _simulate_batch(batch_index, iterations):
    rng = np.random.default_rng(_batch_seed(batch_index))
{indent(distribution_lines, '    ')}

    branch_values = []
    with np.errstate(all="ignore"):
{indent(result_lines, '        ')}

    branch_results = []
    for branch_index, values in enumerate(branch_values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        valid_results = int(values.size)
        discarded_results = int(iterations - valid_results)
        if valid_results:
            batch_sum = float(np.sum(values, dtype=np.float64))
            batch_sum_sq = float(np.sum(values * values, dtype=np.float64))
            batch_min = float(np.min(values))
            batch_max = float(np.max(values))
            batch_avg = batch_sum / valid_results
        else:
            batch_sum = 0.0
            batch_sum_sq = 0.0
            batch_min = None
            batch_max = None
            batch_avg = None

        branch_results.append({{
            "id": BRANCH_IDS[branch_index],
            "label": BRANCH_LABELS[branch_index],
            "values": values,
            "valid_results": valid_results,
            "discarded_results": discarded_results,
            "sum": batch_sum,
            "sum_sq": batch_sum_sq,
            "avg": batch_avg,
            "min": batch_min,
            "max": batch_max,
        }})

    return {{
        "batch_index": int(batch_index),
        "iterations": int(iterations),
        "branches": branch_results,
    }}


def _batch_plan():
    batches = []
    remaining = TOTAL_ITERATIONS
    batch_index = 0
    while remaining > 0:
        size = min(BATCH_SIZE, remaining)
        batches.append((batch_index, size))
        batch_index += 1
        remaining -= size
    return batches


def _new_branch_state(branch_id, label):
    return {{
        "id": branch_id,
        "label": label,
        "all_chunks": [],
        "batch_sums": {{}},
        "batch_sums_sq": {{}},
        "valid_results": 0,
        "discarded_results": 0,
        "global_min": None,
        "global_max": None,
        "histogram_counts": None,
        "histogram_edges": None,
    }}


def _branch_payload(state, batch_result=None):
    valid = state["valid_results"]
    total_sum = math.fsum(state["batch_sums"][index] for index in sorted(state["batch_sums"]))
    total_sum_sq = math.fsum(
        state["batch_sums_sq"][index] for index in sorted(state["batch_sums_sq"])
    )
    average = total_sum / valid if valid else None
    payload = {{
        "id": state["id"],
        "label": state["label"],
        "valid_results": valid,
        "discarded_results": state["discarded_results"],
        "stats": _stats_payload(average, state["global_min"], state["global_max"]),
        "diagnostics": _numerical_diagnostics(valid, total_sum, total_sum_sq),
        "histogram": _histogram_payload(
            state["histogram_counts"],
            state["histogram_edges"],
            state["global_min"],
            state["global_max"],
        ),
    }}
    if batch_result is not None:
        payload["batch"] = {{
            "valid_results": batch_result["valid_results"],
            "discarded_results": batch_result["discarded_results"],
            "stats": _stats_payload(
                batch_result["avg"], batch_result["min"], batch_result["max"]
            ),
        }}
    return payload


def _run():
    batches = _batch_plan()
    worker_count = min(MAX_WORKERS, len(batches))

    _emit({{
        "type": "start",
        "protocol_version": PROTOCOL_VERSION,
        "seed": str(MASTER_SEED),
        "iterations": TOTAL_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "batch_count": len(batches),
        "workers": worker_count,
        "histogram_bins": HISTOGRAM_BINS,
        "requested_statistics": list(REQUESTED_STATISTICS),
        "branches": [
            {{"id": branch_id, "label": label}}
            for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
        ],
        "has_plus_minus": BRANCH_COUNT == 2,
    }})

    branch_states = {{
        branch_id: _new_branch_state(branch_id, label)
        for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
    }}
    processed_iterations = 0
    completed_batches = 0

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {{
            executor.submit(_simulate_batch, batch_index, size): batch_index
            for batch_index, size in batches
        }}

        for future in as_completed(futures):
            result = future.result()
            completed_batches += 1
            processed_iterations += result["iterations"]
            batch_branch_lookup = {{branch["id"]: branch for branch in result["branches"]}}

            for branch_result in result["branches"]:
                state = branch_states[branch_result["id"]]
                state["valid_results"] += branch_result["valid_results"]
                state["discarded_results"] += branch_result["discarded_results"]
                state["batch_sums"][result["batch_index"]] = branch_result["sum"]
                state["batch_sums_sq"][result["batch_index"]] = branch_result["sum_sq"]

                values = branch_result["values"]
                previous_min = state["global_min"]
                previous_max = state["global_max"]

                if branch_result["valid_results"]:
                    state["all_chunks"].append(values)
                    batch_min = branch_result["min"]
                    batch_max = branch_result["max"]
                    state["global_min"] = (
                        batch_min if state["global_min"] is None else min(state["global_min"], batch_min)
                    )
                    state["global_max"] = (
                        batch_max if state["global_max"] is None else max(state["global_max"], batch_max)
                    )

                    range_changed = (
                        state["global_min"] != previous_min or state["global_max"] != previous_max
                    )
                    if state["histogram_counts"] is None or range_changed:
                        state["histogram_counts"], state["histogram_edges"] = _recompute_histogram(
                            state["all_chunks"], state["global_min"], state["global_max"]
                        )
                    else:
                        new_counts, _ = np.histogram(values, bins=state["histogram_edges"])
                        state["histogram_counts"] += new_counts.astype(np.int64, copy=False)

            branches_payload = [
                _branch_payload(state, batch_branch_lookup[state["id"]])
                for state in branch_states.values()
            ]
            valid_total = sum(branch["valid_results"] for branch in branches_payload)
            discarded_total = sum(branch["discarded_results"] for branch in branches_payload)

            _emit({{
                "type": "batch",
                "batch_index": result["batch_index"],
                "completed_batches": completed_batches,
                "batch_count": len(batches),
                "processed_iterations": processed_iterations,
                "total_iterations": TOTAL_ITERATIONS,
                "progress": processed_iterations / TOTAL_ITERATIONS,
                "candidate_results": processed_iterations * BRANCH_COUNT,
                "valid_results": valid_total,
                "discarded_results": discarded_total,
                "batch": {{"iterations": result["iterations"]}},
                "branches": branches_payload,
                "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
                "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
                "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
                "done": False,
            }})

    branches_payload = [_branch_payload(state) for state in branch_states.values()]
    valid_total = sum(branch["valid_results"] for branch in branches_payload)
    discarded_total = sum(branch["discarded_results"] for branch in branches_payload)
    if valid_total == 0:
        raise RuntimeError("La simulacion termino sin ningun resultado numerico finito.")

    _emit({{
        "type": "complete",
        "processed_iterations": processed_iterations,
        "total_iterations": TOTAL_ITERATIONS,
        "progress": 1.0,
        "candidate_results": TOTAL_ITERATIONS * BRANCH_COUNT,
        "valid_results": valid_total,
        "discarded_results": discarded_total,
        "branches": branches_payload,
        "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
        "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
        "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
        "done": True,
    }})


def main():
    multiprocessing.freeze_support()
    try:
        _run()
    except Exception as exc:
        _emit({{
            "type": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "done": True,
        }})
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
        return dedent(template)

    def _distribution_lines(self) -> str:
        lines: list[str] = []
        for distribution in self.program.ast.distributions:
            source_name = distribution.variable
            name = self.variable_names[source_name]
            if isinstance(distribution, NormalDistributionNode):
                mean = self._decimal(distribution.mean, distribution.span)
                stddev = self._decimal(distribution.stddev, distribution.span)
                lines.append(
                    f"{name} = rng.normal(loc={mean}, scale={stddev}, size=iterations).astype(np.float64, copy=False)"
                )
                continue

            if isinstance(distribution, UniformDistributionNode):
                minimum_value = NumpyExpressionVisitor._float64(
                    distribution.minimum, distribution.span
                )
                maximum_value = NumpyExpressionVisitor._float64(
                    distribution.maximum, distribution.span
                )
                if not minimum_value < maximum_value:
                    self._error(
                        "GEN005",
                        (
                            f"Los limites de \\uniform para '{source_name}' dejan de ser "
                            "distintos y ordenados al convertirse a float64."
                        ),
                        distribution.span,
                    )
                minimum = repr(minimum_value)
                maximum = repr(maximum_value)
                lines.append(
                    f"{name} = rng.uniform(low={minimum}, high={maximum}, size=iterations).astype(np.float64, copy=False)"
                )
                continue

            if isinstance(distribution, DiscreteDistributionNode):
                converted_values: list[float] = []
                source_values: dict[float, Decimal] = {}
                for pair in distribution.pairs:
                    converted = NumpyExpressionVisitor._float64(pair.value, pair.span)
                    previous = source_values.get(converted)
                    if previous is not None and previous != pair.value:
                        self._error(
                            "GEN006",
                            (
                                f"Dos valores distintos de \\distrib para '{source_name}' "
                                "se vuelven indistinguibles al convertirse a float64."
                            ),
                            pair.span,
                        )
                    source_values[converted] = pair.value
                    converted_values.append(converted)

                values = ", ".join(repr(value) for value in converted_values)
                probabilities = ", ".join(
                    self._decimal(pair.probability, pair.span) for pair in distribution.pairs
                )
                lines.append(
                    f"{name} = rng.choice(np.array([{values}], dtype=np.float64), "
                    f"size=iterations, p=np.array([{probabilities}], dtype=np.float64))"
                )
                continue

            raise TypeError(f"Unsupported distribution node: {type(distribution).__name__}")

        return "\n".join(lines)

    @staticmethod
    def _result_lines(expression: ExpressionCode) -> str:
        lines: list[str] = []
        for index, branch in enumerate(expression.branches):
            lines.extend(
                [
                    f"_result_{index} = np.asarray({branch}, dtype=np.float64).reshape(-1)",
                    f"_finite_{index} = _result_{index}[np.isfinite(_result_{index})]",
                    f"branch_values.append(_finite_{index})",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _decimal(value: Decimal, span: SourceSpan) -> str:
        return NumpyExpressionVisitor._decimal(value, span)

    @staticmethod
    def _error(code: str, message: str, span: SourceSpan) -> None:
        raise CompilationError(
            Diagnostic(
                code=code,
                phase=DiagnosticPhase.GENERATION,
                message=message,
                span=span,
            )
        )


def generate_python_script(
    program: ValidatedProgram,
    config: GeneratorConfig | None = None,
) -> str:
    return PythonCodeGenerator(program, config).generate()


def generate_source(
    source: str,
    config: GeneratorConfig | None = None,
) -> str:
    return generate_python_script(analyze_source(source), config)
