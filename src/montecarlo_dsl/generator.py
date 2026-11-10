from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import inspect
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
    Statistic,
    UnaryOperationNode,
    UnaryOperator,
    UniformDistributionNode,
    VariableReferenceNode,
)
from .cost_model import (
    ASTCostFeatures,
    InterpretableASTCostModel,
    extract_ast_cost_features,
)
from .diagnostics import CompilationError, Diagnostic, DiagnosticPhase
from .optimizer import OptimizationMetrics, optimize_program
from .scheduler import (
    AdaptiveMonteCarloScheduler,
    ExecutionSchedulePlan,
    VALID_SCHEDULER_MODES,
    build_runtime_schedule_plan,
    count_runtime_schedule_tasks,
    iter_runtime_schedule_tasks,
    normalize_scheduler_mode,
)
from .semantic import ValidatedProgram, analyze_source
from .source import SourceSpan
from .statistics import (
    BlockHistogramSummary,
    BoundedStreamingHistogram,
    compact_weighted_sketch,
    weighted_quantiles,
    wilson_interval,
)


VALID_SEED_STRATEGIES = ("philox_logical_block", "batch_index")
VALID_MEMORY_MODES = ("bounded", "retain_all")
VALID_COMPLETION_ORDERS = ("as_completed", "forward", "reverse", "shuffled", "delayed_first")

_SHARED_STATISTICS_SOURCE = "\n\n".join(
    inspect.getsource(obj).rstrip()
    for obj in (
        weighted_quantiles,
        compact_weighted_sketch,
        BlockHistogramSummary,
        BoundedStreamingHistogram,
        wilson_interval,
    )
)
_SHARED_SCHEDULER_SOURCE = "\n\n".join(
    inspect.getsource(obj).rstrip()
    for obj in (iter_runtime_schedule_tasks, count_runtime_schedule_tasks, build_runtime_schedule_plan)
)


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Configuracion del script Monte Carlo generado (v1.3.1)."""

    batch_size: int = 25_000
    max_workers: int = 4
    histogram_bins: int = 30
    seed: int | None = None
    logical_block_size: int = 5_000
    scheduler_mode: str = "hybrid"
    seed_strategy: str = "philox_logical_block"
    memory_mode: str = "bounded"
    optimize: bool = True
    task_completion_order: str = "as_completed"
    # Experimental stress hook. Zero has no effect on normal executions.
    # When positive, only physical task 0 sleeps before computing its payload.
    task_delay_first_seconds: float = 0.0
    max_iterations: int = 1_000_000

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be greater than zero")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.logical_block_size <= 0:
            raise ValueError("logical_block_size must be greater than zero")
        if self.scheduler_mode not in VALID_SCHEDULER_MODES:
            raise ValueError(f"scheduler_mode must be one of {VALID_SCHEDULER_MODES}")
        if self.seed_strategy not in VALID_SEED_STRATEGIES:
            raise ValueError(f"seed_strategy must be one of {VALID_SEED_STRATEGIES}")
        if self.memory_mode not in VALID_MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {VALID_MEMORY_MODES}")
        if self.task_completion_order not in VALID_COMPLETION_ORDERS:
            raise ValueError(f"task_completion_order must be one of {VALID_COMPLETION_ORDERS}")
        try:
            task_delay = float(self.task_delay_first_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("task_delay_first_seconds must be a finite non-negative number") from exc
        if not math.isfinite(task_delay) or task_delay < 0:
            raise ValueError("task_delay_first_seconds must be a finite non-negative number")
        if self.task_completion_order == "delayed_first" and task_delay <= 0:
            raise ValueError("delayed_first requires task_delay_first_seconds > 0")


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
            return self._combine(base, exponent, lambda a, b: f"np.power(({a}), ({b}))")

        if isinstance(node, FractionNode):
            numerator = self.visit(node.numerator)
            denominator = self.visit(node.denominator)
            return self._combine(numerator, denominator, lambda a, b: f"(({a}) / ({b}))")

        if isinstance(node, SqrtNode):
            expression = self.visit(node.expression)
            return ExpressionCode(tuple(f"np.sqrt(({branch}))" for branch in expression.branches))

        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    @staticmethod
    def _combine(left: ExpressionCode, right: ExpressionCode, render) -> ExpressionCode:
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
            Diagnostic(code=code, phase=DiagnosticPhase.GENERATION, message=message, span=span)
        )


class PythonCodeGenerator:
    """Genera un script Python/NumPy autocontenido desde ``ValidatedProgram``."""

    def __init__(
        self,
        program: ValidatedProgram,
        config: GeneratorConfig | None = None,
        cost_model: InterpretableASTCostModel | None = None,
        static_schedule_plan: ExecutionSchedulePlan | None = None,
    ):
        self.raw_program = program
        self.config = config or GeneratorConfig()
        self.cost_model = cost_model or InterpretableASTCostModel()
        self.program, self.optimization_metrics = optimize_program(
            program,
            enabled=self.config.optimize,
        )
        self.cost_features: ASTCostFeatures = extract_ast_cost_features(self.program)
        self.static_per_sample_seconds: float = (
            self.cost_model.estimate_static_per_sample_seconds(self.cost_features)
        )
        self.scheduler = AdaptiveMonteCarloScheduler(self.cost_model)
        self.static_schedule_plan: ExecutionSchedulePlan = (
            static_schedule_plan
            or self.scheduler.build_plan(
                self.cost_features,
                total_iterations=self.program.iterations,
                batch_size=self.config.batch_size,
                logical_block_size=self.config.logical_block_size,
                max_workers=self.config.max_workers,
                scheduler_mode=self.config.scheduler_mode,
            )
        )
        self.seed = self.config.seed if self.config.seed is not None else secrets.randbits(128)
        self.variable_names = {
            name: f"_mc_var_{index}" for index, name in enumerate(self.program.random_variables)
        }
        self.expression_visitor = NumpyExpressionVisitor(self.variable_names)

    def generate(self) -> str:
        _ = self.expression_visitor.visit(self.raw_program.ast.model.expression)
        expression = self.expression_visitor.visit(self.program.ast.model.expression)
        expected_branches = 2 if self.program.has_plus_minus else 1
        if len(expression.branches) != expected_branches:
            self._error(
                "GEN003",
                "La cantidad de ramas generadas no coincide con el resultado del analisis semantico.",
                self.program.ast.model.expression.span,
            )

        distribution_lines_philox = self._distribution_lines(use_philox_per_var=True)
        distribution_lines_legacy = self._distribution_lines(use_philox_per_var=False)
        result_lines = self._result_lines(expression)
        requested_tuple = tuple(item.value for item in self.program.requested_statistics)
        requested = repr(requested_tuple)
        requires_quantiles = repr(
            any(
                item in {Statistic.P05, Statistic.P50, Statistic.P95}
                for item in self.program.requested_statistics
            )
        )
        branch_ids = ("plus", "minus") if len(expression.branches) == 2 else ("main",)
        branch_labels = ("Rama +", "Rama -") if len(expression.branches) == 2 else ("Resultado",)
        normalized_mode = normalize_scheduler_mode(self.config.scheduler_mode)

        template = '''\
# Generado por montecarlo_dsl v1.3.1.
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
import math
import multiprocessing
import os
import platform
import random
import sys
import time
import traceback
from typing import Any, Iterable, Sequence

try:
    import resource
except ImportError:  # Windows
    resource = None

import numpy as np

PROTOCOL_VERSION = 2
STATISTICAL_CORE_VERSION = "1.3.1"
RUNTIME_VERSION = "1.3.1"
EVENT_PREFIX = "MC_DSL_EVENT:"
MASTER_SEED = __MASTER_SEED__
TOTAL_ITERATIONS = __TOTAL_ITERATIONS__
BATCH_SIZE = __BATCH_SIZE__
LOGICAL_BLOCK_SIZE = __LOGICAL_BLOCK_SIZE__
EFF_LOGICAL_BLOCK_SIZE = min(LOGICAL_BLOCK_SIZE, TOTAL_ITERATIONS)
LOGICAL_BLOCK_COUNT = (TOTAL_ITERATIONS + EFF_LOGICAL_BLOCK_SIZE - 1) // EFF_LOGICAL_BLOCK_SIZE
MAX_WORKERS = __MAX_WORKERS__
HISTOGRAM_BINS = __HISTOGRAM_BINS__
SCHEDULER_MODE = __SCHEDULER_MODE__
SEED_STRATEGY = __SEED_STRATEGY__
MEMORY_MODE = __MEMORY_MODE__
TASK_COMPLETION_ORDER = __TASK_COMPLETION_ORDER__
TASK_DELAY_FIRST_SECONDS = __TASK_DELAY_FIRST_SECONDS__
OPTIMIZER_ENABLED = __OPTIMIZER_ENABLED__
OPTIMIZATION_METRICS = __OPTIMIZATION_METRICS__
AST_COST_FEATURES = __AST_COST_FEATURES__
COST_MODEL_WEIGHTS = __COST_MODEL_WEIGHTS__
PRIOR_COST_MODEL_WEIGHTS = __PRIOR_COST_MODEL_WEIGHTS__
STATIC_PER_SAMPLE_SECONDS = __STATIC_PER_SAMPLE_SECONDS__
SERIAL_TASK_OVERHEAD_SECONDS = __SERIAL_TASK_OVERHEAD_SECONDS__
PARALLEL_POOL_SPAWN_SECONDS = __PARALLEL_POOL_SPAWN_SECONDS__
PARALLEL_TASK_IPC_SECONDS = __PARALLEL_TASK_IPC_SECONDS__
PRIOR_EQUIVALENT_SAMPLES = __PRIOR_EQUIVALENT_SAMPLES__
ESTIMATED_BYTES_PER_SAMPLE = __ESTIMATED_BYTES_PER_SAMPLE__
STATIC_TASK_GENERATION = __STATIC_TASK_GENERATION__
STATIC_TASK_COUNT = __STATIC_TASK_COUNT__
STATIC_EFFECTIVE_WORKERS = __STATIC_EFFECTIVE_WORKERS__
STATIC_USE_PROCESS_POOL = __STATIC_USE_PROCESS_POOL__
REQUESTED_STATISTICS = __REQUESTED_STATISTICS__
REQUIRES_QUANTILES = __REQUIRES_QUANTILES__
BRANCH_IDS = __BRANCH_IDS__
BRANCH_LABELS = __BRANCH_LABELS__
BRANCH_COUNT = len(BRANCH_IDS)
EXACT_SAMPLE_RETENTION_LIMIT = 65_536
FINE_HISTOGRAM_BINS = 256
MAX_GLOBAL_SKETCH_POINTS = 2048
MAX_PENDING_LOGICAL_BLOCKS = 64
MAX_IN_FLIGHT_TASKS_PER_WORKER = 2

# Implementacion unificada inyectada directamente desde montecarlo_dsl.statistics
__SHARED_STATISTICS_SOURCE__

__SHARED_SCHEDULER_SOURCE__


def _process_peak_rss_bytes():
    """Return per-process peak working set/RSS in bytes where the OS exposes it."""
    if resource is not None:
        try:
            ru = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return ru if sys.platform == "darwin" else ru * 1024
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process_handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(process_handle, ctypes.byref(counters), counters.cb)
            if ok:
                return int(counters.PeakWorkingSetSize)
        except Exception:
            pass
    return 0


def _emit(payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(EVENT_PREFIX + encoded, flush=True)


def _batch_seed_sequence(batch_index):
    return np.random.SeedSequence([MASTER_SEED, int(batch_index)])


def _variable_rng(logical_block_index, var_index):
    ss = np.random.SeedSequence([MASTER_SEED, int(logical_block_index), int(var_index)])
    return np.random.default_rng(np.random.Philox(ss))


def _empty_stats_state():
    return {"n": 0, "mean": 0.0, "m2": 0.0, "min": None, "max": None}


def _stats_from_values(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(values.size)
    if n == 0:
        return _empty_stats_state()
    mean = float(np.mean(values, dtype=np.float64))
    centered = values - mean
    m2 = float(np.sum(centered * centered, dtype=np.float64))
    return {
        "n": n,
        "mean": mean,
        "m2": m2,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _merge_stats(left, right):
    if right["n"] == 0:
        return dict(left)
    if left["n"] == 0:
        return dict(right)
    n_left = int(left["n"])
    n_right = int(right["n"])
    n = n_left + n_right
    delta = right["mean"] - left["mean"]
    mean = left["mean"] + delta * n_right / n
    m2 = left["m2"] + right["m2"] + delta * delta * n_left * n_right / n
    return {
        "n": n,
        "mean": mean,
        "m2": m2,
        "min": min(left["min"], right["min"]),
        "max": max(left["max"], right["max"]),
    }


def _stats_from_state(state):
    merged = state["canonical_stats"]
    for key in sorted(state["batch_stats"]):
        merged = _merge_stats(merged, state["batch_stats"][key])
    return merged


def _sample_variance(stats_state):
    n = stats_state["n"]
    if n <= 1:
        return None
    return max(0.0, stats_state["m2"] / (n - 1))


def _sample_stddev(stats_state):
    variance = _sample_variance(stats_state)
    return None if variance is None else math.sqrt(variance)


def _mcse_mean(stats_state):
    stddev = _sample_stddev(stats_state)
    if stddev is None or stats_state["n"] <= 0:
        return None
    return stddev / math.sqrt(stats_state["n"])


def _ci95_mean(stats_state):
    mcse = _mcse_mean(stats_state)
    if mcse is None:
        return None
    mean = stats_state["mean"]
    return [mean - 1.96 * mcse, mean + 1.96 * mcse]


def _wilson_interval(successes, trials, z=1.96):
    return wilson_interval(successes, trials, z=z)


def _weighted_quantiles(centers, weights):
    q = weighted_quantiles(centers, weights, (0.05, 0.50, 0.95))
    if q is None:
        return {"p05": None, "p50": None, "p95": None}
    return {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def _compact_weighted_sketch(centers, weights, data_min, data_max, target_bins=MAX_GLOBAL_SKETCH_POINTS):
    return compact_weighted_sketch(
        centers,
        weights,
        data_min,
        data_max,
        target_bins=target_bins,
    )


def _summarize_block_values(values, stats_state):
    n = int(stats_state["n"])
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "count": 0,
            "min": None,
            "max": None,
            "centers": empty,
            "weights": np.empty(0, dtype=np.int64),
        }
    summary_obj = BlockHistogramSummary.from_array(
        values,
        fine_bins=FINE_HISTOGRAM_BINS,
        max_quantile_samples=FINE_HISTOGRAM_BINS,
        precomputed_min=stats_state["min"],
        precomputed_max=stats_state["max"],
    )
    return summary_obj.to_dict()


def _use_compact_block_summaries():
    return MEMORY_MODE == "bounded" and TOTAL_ITERATIONS > EXACT_SAMPLE_RETENTION_LIMIT


def _make_streaming_histogram():
    return BoundedStreamingHistogram(
        bins=HISTOGRAM_BINS,
        fine_bins=FINE_HISTOGRAM_BINS,
        max_quantile_samples_per_block=FINE_HISTOGRAM_BINS,
        max_global_points=MAX_GLOBAL_SKETCH_POINTS,
        exact_retention=not _use_compact_block_summaries(),
    )


def _quantile_payload(sketch, compute_now=True):
    if not REQUIRES_QUANTILES or not compute_now or sketch is None:
        return {"p05": None, "p50": None, "p95": None}
    return sketch.quantiles((0.05, 0.50, 0.95))


def _stats_payload(stats_state, candidate_results, sketch, compute_quantiles=True):
    n = int(stats_state["n"])
    variance = _sample_variance(stats_state)
    stddev = _sample_stddev(stats_state)
    quantiles = _quantile_payload(sketch, compute_now=compute_quantiles)
    valid_rate = n / candidate_results if candidate_results else None
    discard_rate = 1.0 - valid_rate if valid_rate is not None else None
    available = {
        "avg": stats_state["mean"] if n else None,
        "min": stats_state["min"],
        "max": stats_state["max"],
        "var": variance,
        "std": stddev,
        "count": n,
        "valid_rate": valid_rate,
        "discard_rate": discard_rate,
        "p05": quantiles["p05"],
        "p50": quantiles["p50"],
        "p95": quantiles["p95"],
    }
    return {name: available[name] for name in REQUESTED_STATISTICS}


def _numerical_diagnostics(stats_state, candidate_results):
    n = int(stats_state["n"])
    discarded = max(0, int(candidate_results) - n)
    valid_rate = n / candidate_results if candidate_results else None
    discard_rate = discarded / candidate_results if candidate_results else None
    stddev = _sample_stddev(stats_state)
    mcse = _mcse_mean(stats_state)
    ci95 = _ci95_mean(stats_state)
    relative = abs(mcse / stats_state["mean"]) if mcse is not None and stats_state["mean"] != 0 else None
    return {
        "n": n,
        "candidate_results": int(candidate_results),
        "sample_min": stats_state["min"],
        "sample_max": stats_state["max"],
        "sample_variance": _sample_variance(stats_state),
        "sample_stddev": stddev,
        "mcse_mean": mcse,
        "ci95_mean": ci95,
        "ci95_mean_type": "normal_approximation",
        "relative_mcse": relative,
        "valid_rate": valid_rate,
        "discard_rate": discard_rate,
        "valid_rate_ci95_wilson": _wilson_interval(n, candidate_results),
        "discard_rate_ci95_wilson": _wilson_interval(discarded, candidate_results),
        "nonfinite_policy": "discard_and_report",
    }


def _histogram_payload(counts, edges, data_min, data_max):
    if counts is None or edges is None or data_min is None or data_max is None:
        return {"data_min": None, "data_max": None, "total_count": 0, "bins": []}
    bins = [
        {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]
    return {
        "data_min": float(data_min),
        "data_max": float(data_max),
        "total_count": int(np.sum(counts, dtype=np.int64)),
        "bins": bins,
    }


def _coerce_result_array(values, iterations):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == iterations:
        return array
    if array.size == 1:
        return np.full(iterations, float(array[0]), dtype=np.float64)
    raise ValueError(
        f"La expresion del modelo produjo {array.size} valores para {iterations} iteraciones."
    )


def _evaluate_variables_philox(logical_block_index, iterations):
__DISTRIBUTION_LINES_PHILOX__
    branch_values = []
    branch_raw_values = []
    with np.errstate(all="ignore"):
__RESULT_LINES__
    return branch_raw_values, branch_values


def _evaluate_variables_legacy(batch_index, iterations):
    rng = np.random.default_rng(_batch_seed_sequence(batch_index))
__DISTRIBUTION_LINES_LEGACY__
    branch_values = []
    branch_raw_values = []
    with np.errstate(all="ignore"):
__RESULT_LINES__
    return branch_raw_values, branch_values


def _build_branch_and_paired_records(branch_raw_values, branch_values, iterations, compact_summary=False, compute_batch_quantiles=True):
    branch_results = []
    for branch_index, values in enumerate(branch_values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        valid_results = int(values.size)
        discarded_results = int(iterations - valid_results)
        stats_state = _stats_from_values(values)
        summary = _summarize_block_values(values, stats_state) if compact_summary else None
        if REQUIRES_QUANTILES and compute_batch_quantiles:
            if summary is not None:
                batch_quantiles = _weighted_quantiles(summary["centers"], summary["weights"])
            elif valid_results > 0:
                q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
                batch_quantiles = {"p05": float(q05), "p50": float(q50), "p95": float(q95)}
            else:
                batch_quantiles = {"p05": None, "p50": None, "p95": None}
        else:
            batch_quantiles = {"p05": None, "p50": None, "p95": None}
        branch_results.append({
            "id": BRANCH_IDS[branch_index],
            "label": BRANCH_LABELS[branch_index],
            "values": None if compact_summary else values,
            "summary": summary,
            "batch_quantiles": batch_quantiles,
            "candidate_results": int(iterations),
            "valid_results": valid_results,
            "discarded_results": discarded_results,
            "stats_state": stats_state,
            "avg": stats_state["mean"] if valid_results else None,
            "min": stats_state["min"],
            "max": stats_state["max"],
        })

    paired = None
    if BRANCH_COUNT == 2:
        mask_plus = np.isfinite(branch_raw_values[0])
        mask_minus = np.isfinite(branch_raw_values[1])
        mask_pair = mask_plus & mask_minus
        delta_values = branch_raw_values[0][mask_pair] - branch_raw_values[1][mask_pair]
        paired_stats = _stats_from_values(delta_values)
        correlation = None
        if paired_stats["n"] > 1:
            plus_pair = branch_raw_values[0][mask_pair]
            minus_pair = branch_raw_values[1][mask_pair]
            corr_matrix = np.corrcoef(plus_pair, minus_pair)
            candidate_corr = float(corr_matrix[0, 1])
            correlation = candidate_corr if math.isfinite(candidate_corr) else None
        summary = _summarize_block_values(delta_values, paired_stats) if compact_summary else None
        if REQUIRES_QUANTILES and compute_batch_quantiles:
            if summary is not None:
                batch_quantiles = _weighted_quantiles(summary["centers"], summary["weights"])
            elif delta_values.size > 0:
                q05, q50, q95 = np.quantile(delta_values, [0.05, 0.50, 0.95])
                batch_quantiles = {"p05": float(q05), "p50": float(q50), "p95": float(q95)}
            else:
                batch_quantiles = {"p05": None, "p50": None, "p95": None}
        else:
            batch_quantiles = {"p05": None, "p50": None, "p95": None}
        paired = {
            "id": "plus_minus_delta",
            "label": "Diferencia pareada: Rama + menos Rama -",
            "candidate_results": int(iterations),
            "valid_results": int(delta_values.size),
            "discarded_results": int(iterations - delta_values.size),
            "stats_state": paired_stats,
            "correlation_plus_minus": correlation,
            "values": None if compact_summary else delta_values,
            "summary": summary,
            "batch_quantiles": batch_quantiles,
        }
    return branch_results, paired


def _merge_logical_blocks_into_task_summary(logical_blocks, task_iterations):
    ordered_keys = sorted(logical_blocks)
    branch_results = []
    for b_idx in range(BRANCH_COUNT):
        stats_state = _empty_stats_state()
        valid_total = 0
        c_parts = []
        w_parts = []
        v_parts = []
        for k in ordered_keys:
            lb_b = logical_blocks[k]["branches"][b_idx]
            stats_state = _merge_stats(stats_state, lb_b["stats_state"])
            valid_total += lb_b["valid_results"]
            if REQUIRES_QUANTILES and lb_b["valid_results"] > 0:
                if lb_b["summary"] is not None:
                    c_parts.append(lb_b["summary"]["centers"])
                    w_parts.append(lb_b["summary"]["weights"])
                elif lb_b["values"] is not None:
                    v_parts.append(lb_b["values"])

        if REQUIRES_QUANTILES and valid_total > 0:
            if c_parts:
                batch_quantiles = _weighted_quantiles(np.concatenate(c_parts), np.concatenate(w_parts))
            elif v_parts:
                vals = np.concatenate(v_parts)
                q05, q50, q95 = np.quantile(vals, [0.05, 0.50, 0.95])
                batch_quantiles = {"p05": float(q05), "p50": float(q50), "p95": float(q95)}
            else:
                batch_quantiles = {"p05": None, "p50": None, "p95": None}
        else:
            batch_quantiles = {"p05": None, "p50": None, "p95": None}

        branch_results.append({
            "id": BRANCH_IDS[b_idx],
            "label": BRANCH_LABELS[b_idx],
            "values": None,
            "summary": None,
            "batch_quantiles": batch_quantiles,
            "candidate_results": int(task_iterations),
            "valid_results": int(valid_total),
            "discarded_results": int(task_iterations - valid_total),
            "stats_state": stats_state,
            "avg": stats_state["mean"] if valid_total else None,
            "min": stats_state["min"],
            "max": stats_state["max"],
        })

    paired = None
    if BRANCH_COUNT == 2:
        p_stats = _empty_stats_state()
        p_valid = 0
        corrs = []
        c_parts = []
        w_parts = []
        v_parts = []
        for k in ordered_keys:
            lb_p = logical_blocks[k]["paired"]
            p_stats = _merge_stats(p_stats, lb_p["stats_state"])
            p_valid += lb_p["valid_results"]
            if lb_p["correlation_plus_minus"] is not None:
                corrs.append(lb_p["correlation_plus_minus"])
            if REQUIRES_QUANTILES and lb_p["valid_results"] > 0:
                if lb_p["summary"] is not None:
                    c_parts.append(lb_p["summary"]["centers"])
                    w_parts.append(lb_p["summary"]["weights"])
                elif lb_p["values"] is not None:
                    v_parts.append(lb_p["values"])

        if REQUIRES_QUANTILES and p_valid > 0:
            if c_parts:
                batch_quantiles = _weighted_quantiles(np.concatenate(c_parts), np.concatenate(w_parts))
            elif v_parts:
                vals = np.concatenate(v_parts)
                q05, q50, q95 = np.quantile(vals, [0.05, 0.50, 0.95])
                batch_quantiles = {"p05": float(q05), "p50": float(q50), "p95": float(q95)}
            else:
                batch_quantiles = {"p05": None, "p50": None, "p95": None}
        else:
            batch_quantiles = {"p05": None, "p50": None, "p95": None}

        paired = {
            "id": "plus_minus_delta",
            "label": "Diferencia pareada: Rama + menos Rama -",
            "candidate_results": int(task_iterations),
            "valid_results": int(p_valid),
            "discarded_results": int(task_iterations - p_valid),
            "stats_state": p_stats,
            "correlation_plus_minus": float(math.fsum(corrs) / len(corrs)) if corrs else None,
            "values": None,
            "summary": None,
            "batch_quantiles": batch_quantiles,
        }
    return branch_results, paired


def _simulate_batch(batch_index, iterations, start_sample=None, end_sample=None):
    # Controlled experiment-only stress hook. It is disabled by default and
    # does not alter scheduler decisions or the random-number stream.
    if int(batch_index) == 0 and TASK_DELAY_FIRST_SECONDS > 0:
        time.sleep(TASK_DELAY_FIRST_SECONDS)
    compact = _use_compact_block_summaries()
    if SEED_STRATEGY == "batch_index":
        raw_vals, fin_vals = _evaluate_variables_legacy(batch_index, iterations)
        branch_results, paired = _build_branch_and_paired_records(
            raw_vals, fin_vals, iterations, compact_summary=compact
        )
        return {
            "batch_index": int(batch_index),
            "iterations": int(iterations),
            "start_sample": int(batch_index) * int(iterations),
            "end_sample": (int(batch_index) + 1) * int(iterations),
            "aligned": True,
            "worker_pid": os.getpid(),
            "worker_peak_rss_bytes": _process_peak_rss_bytes(),
            "branches": branch_results,
            "paired": paired,
            "logical_blocks": {
                int(batch_index): {
                    "branches": branch_results,
                    "paired": paired,
                }
            },
        }

    if start_sample is None or end_sample is None:
        start_sample = int(batch_index) * BATCH_SIZE
        end_sample = min(TOTAL_ITERATIONS, start_sample + int(iterations))
    task_iterations = int(end_sample - start_sample)
    first_block = start_sample // EFF_LOGICAL_BLOCK_SIZE
    last_block = (end_sample - 1) // EFF_LOGICAL_BLOCK_SIZE

    aligned = (
        start_sample == first_block * EFF_LOGICAL_BLOCK_SIZE
        and end_sample == min(TOTAL_ITERATIONS, (last_block + 1) * EFF_LOGICAL_BLOCK_SIZE)
    )

    logical_blocks = {}

    if aligned:
        single_block = (first_block == last_block)
        for b in range(first_block, last_block + 1):
            b_start = b * EFF_LOGICAL_BLOCK_SIZE
            b_end = min(TOTAL_ITERATIONS, (b + 1) * EFF_LOGICAL_BLOCK_SIZE)
            b_iters = b_end - b_start
            b_raw, b_fin = _evaluate_variables_philox(b, b_iters)
            lb_branches, lb_paired = _build_branch_and_paired_records(
                b_raw,
                b_fin,
                b_iters,
                compact_summary=compact,
                compute_batch_quantiles=single_block,
            )
            logical_blocks[int(b)] = {
                "branches": lb_branches,
                "paired": lb_paired,
            }

        if single_block:
            only_lb = logical_blocks[int(first_block)]
            return {
                "batch_index": int(batch_index),
                "iterations": task_iterations,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "aligned": True,
                "worker_pid": os.getpid(),
                "worker_peak_rss_bytes": _process_peak_rss_bytes(),
                "branches": only_lb["branches"],
                "paired": only_lb["paired"],
                "logical_blocks": logical_blocks,
            }

        task_branches, task_paired = _merge_logical_blocks_into_task_summary(
            logical_blocks, task_iterations
        )
        return {
            "batch_index": int(batch_index),
            "iterations": task_iterations,
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "aligned": True,
            "worker_pid": os.getpid(),
            "worker_peak_rss_bytes": _process_peak_rss_bytes(),
            "branches": task_branches,
            "paired": task_paired,
            "logical_blocks": logical_blocks,
        }

    task_raw_per_branch = [[] for _ in range(BRANCH_COUNT)]
    task_fin_per_branch = [[] for _ in range(BRANCH_COUNT)]

    for b in range(first_block, last_block + 1):
        b_start = b * EFF_LOGICAL_BLOCK_SIZE
        b_end = min(TOTAL_ITERATIONS, (b + 1) * EFF_LOGICAL_BLOCK_SIZE)
        b_iters = b_end - b_start
        b_raw, b_fin = _evaluate_variables_philox(b, b_iters)

        if start_sample < b_end <= end_sample:
            lb_branches, lb_paired = _build_branch_and_paired_records(
                b_raw, b_fin, b_iters, compact_summary=compact
            )
            logical_blocks[int(b)] = {
                "branches": lb_branches,
                "paired": lb_paired,
            }

        slice_lo = max(start_sample, b_start) - b_start
        slice_hi = min(end_sample, b_end) - b_start
        for idx in range(BRANCH_COUNT):
            sub_raw = b_raw[idx][slice_lo:slice_hi]
            task_raw_per_branch[idx].append(sub_raw)
            task_fin_per_branch[idx].append(sub_raw[np.isfinite(sub_raw)])

    merged_raw = [np.concatenate(parts) for parts in task_raw_per_branch]
    merged_fin = [np.concatenate(parts) for parts in task_fin_per_branch]
    task_branches, task_paired = _build_branch_and_paired_records(
        merged_raw, merged_fin, task_iterations, compact_summary=compact
    )
    return {
        "batch_index": int(batch_index),
        "iterations": task_iterations,
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "aligned": False,
        "worker_pid": os.getpid(),
        "worker_peak_rss_bytes": _process_peak_rss_bytes(),
        "branches": task_branches,
        "paired": task_paired,
        "logical_blocks": logical_blocks,
    }


def _fixed_batch_plan():
    """Yield fixed-size physical tasks without allocating the complete plan."""
    offset = 0
    batch_index = 0
    while offset < TOTAL_ITERATIONS:
        size = min(BATCH_SIZE, TOTAL_ITERATIONS - offset)
        yield (batch_index, size, offset, offset + size)
        batch_index += 1
        offset += size


def _fixed_batch_count():
    return (TOTAL_ITERATIONS + BATCH_SIZE - 1) // BATCH_SIZE


def _iter_static_schedule_tasks():
    if SCHEDULER_MODE == "fixed":
        yield from _fixed_batch_plan()
    else:
        yield from iter_runtime_schedule_tasks(**STATIC_TASK_GENERATION)


def _iter_tasks_in_submission_order(tasks, window_size):
    """Reorder only bounded windows; never materialize the complete task stream."""
    iterator = iter(tasks)
    window_size = max(1, int(window_size))
    rng = random.Random(MASTER_SEED ^ 0xA5A5)
    first_window = True
    while True:
        window = []
        for _ in range(window_size):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            return
        if TASK_COMPLETION_ORDER == "reverse":
            if first_window:
                yield window[0]
                yield from reversed(window[1:])
            else:
                yield from reversed(window)
        elif TASK_COMPLETION_ORDER == "delayed_first" and first_window:
            # Submit every task in the initial bounded window before task 0.
            # Combined with TASK_DELAY_FIRST_SECONDS this intentionally creates
            # an adversarial head-of-line delay while preserving the fixed-size
            # submission window and canonical result-ingestion path.
            yield from window[1:]
            yield window[0]
        elif TASK_COMPLETION_ORDER == "shuffled":
            if first_window:
                first = window.pop(0)
                rng.shuffle(window)
                yield first
                yield from window
            else:
                rng.shuffle(window)
                yield from window
        else:
            yield from window
        first_window = False


def _plan_after_pilot(pilot_size, pilot_elapsed):
    """Use the same pure planning kernel as AdaptiveMonteCarloScheduler.build_plan()."""
    plan = build_runtime_schedule_plan(
        total_iterations=TOTAL_ITERATIONS,
        batch_size=BATCH_SIZE,
        logical_block_size=LOGICAL_BLOCK_SIZE,
        max_workers=MAX_WORKERS,
        scheduler_mode=SCHEDULER_MODE,
        feature_values=AST_COST_FEATURES,
        static_per_sample_seconds=STATIC_PER_SAMPLE_SECONDS,
        prior_equivalent_samples=PRIOR_EQUIVALENT_SAMPLES,
        serial_task_overhead_seconds=SERIAL_TASK_OVERHEAD_SECONDS,
        parallel_pool_spawn_seconds=PARALLEL_POOL_SPAWN_SECONDS,
        parallel_task_ipc_seconds=PARALLEL_TASK_IPC_SECONDS,
        estimated_bytes_per_sample=ESTIMATED_BYTES_PER_SAMPLE,
        pilot_iterations=pilot_size,
        pilot_elapsed_seconds=pilot_elapsed,
        start_logical_block=1,
        initial_task_index=1,
    )
    return (
        plan["tasks"],
        plan["task_count"],
        plan["effective_workers"],
        plan["use_process_pool"],
        plan["pilot_per_sample_seconds"],
        plan["effective_per_sample_seconds"],
    )


def _neumaier_add(current_sum, current_comp, value):
    val = float(value)
    t = current_sum + val
    if abs(current_sum) >= abs(val):
        current_comp += (current_sum - t) + val
    else:
        current_comp += (val - t) + current_sum
    return t, current_comp


def _new_branch_state(branch_id, label):
    return {
        "id": branch_id,
        "label": label,
        "progress_sketch": None,
        "progress_step": 0,
        "canonical_sketch": _make_streaming_histogram(),
        "next_canonical_block": 0,
        "canonical_stats": _empty_stats_state(),
        "candidate_results": 0,
        "valid_results": 0,
        "discarded_results": 0,
        "stats_state": _empty_stats_state(),
        "batch_stats": {},
        "histogram_counts": None,
        "histogram_edges": None,
    }


def _new_paired_state():
    return {
        "id": "plus_minus_delta",
        "label": "Diferencia pareada: Rama + menos Rama -",
        "progress_sketch": None,
        "progress_step": 0,
        "progress_corr_sum": 0.0,
        "progress_corr_comp": 0.0,
        "progress_corr_count": 0,
        "canonical_sketch": _make_streaming_histogram(),
        "next_canonical_block": 0,
        "canonical_stats": _empty_stats_state(),
        "candidate_results": 0,
        "valid_results": 0,
        "discarded_results": 0,
        "stats_state": _empty_stats_state(),
        "batch_stats": {},
        "correlations": {},
        "corr_sum": 0.0,
        "corr_comp": 0.0,
        "corr_count": 0,
        "next_corr_block": 0,
        "histogram_counts": None,
        "histogram_edges": None,
    }


def _advance_state_frontier(state):
    """Avanza la frontera contigua de bloques logicos y libera los bloques ya reducidos (O(1) memoria)."""
    if MEMORY_MODE != "bounded":
        return
    while state["next_canonical_block"] in state["batch_stats"]:
        b_idx = state["next_canonical_block"]
        b_stats = state["batch_stats"].pop(b_idx)
        state["canonical_stats"] = _merge_stats(state["canonical_stats"], b_stats)
        state["next_canonical_block"] = b_idx + 1


def _advance_paired_frontier(paired_state):
    """Avanza la frontera contigua de estadisticos y correlaciones pareadas, liberando cada entrada procesada (O(1))."""
    if paired_state is None:
        return
    _advance_state_frontier(paired_state)
    if MEMORY_MODE != "bounded":
        return
    while paired_state["next_corr_block"] in paired_state["correlations"]:
        c_idx = paired_state["next_corr_block"]
        corr_val = paired_state["correlations"].pop(c_idx)
        if corr_val is not None:
            paired_state["corr_sum"], paired_state["corr_comp"] = _neumaier_add(
                paired_state["corr_sum"],
                paired_state["corr_comp"],
                corr_val,
            )
            paired_state["corr_count"] += 1
        paired_state["next_corr_block"] = c_idx + 1


def _canonical_correlation_mean(paired_state):
    if paired_state is None:
        return None
    c_sum = float(paired_state["corr_sum"])
    c_comp = float(paired_state["corr_comp"])
    c_count = int(paired_state["corr_count"])
    for k in sorted(paired_state["correlations"]):
        val = paired_state["correlations"][k]
        if val is not None:
            c_sum, c_comp = _neumaier_add(c_sum, c_comp, val)
            c_count += 1
    if c_count <= 0:
        return None
    return float((c_sum + c_comp) / c_count)


def _progress_correlation_mean(paired_state):
    if paired_state is None or paired_state["progress_corr_count"] <= 0:
        return None
    return float(
        (paired_state["progress_corr_sum"] + paired_state["progress_corr_comp"])
        / paired_state["progress_corr_count"]
    )


def _batch_item_stats(batch_result):
    n = int(batch_result["stats_state"]["n"])
    cand = int(batch_result["candidate_results"])
    variance = _sample_variance(batch_result["stats_state"])
    stddev = _sample_stddev(batch_result["stats_state"])
    quantiles = batch_result.get("batch_quantiles") or {"p05": None, "p50": None, "p95": None}
    valid_rate = n / cand if cand else None
    discard_rate = 1.0 - valid_rate if valid_rate is not None else None
    available = {
        "avg": batch_result["stats_state"]["mean"] if n else None,
        "min": batch_result["stats_state"]["min"],
        "max": batch_result["stats_state"]["max"],
        "var": variance,
        "std": stddev,
        "count": n,
        "valid_rate": valid_rate,
        "discard_rate": discard_rate,
        "p05": quantiles["p05"],
        "p50": quantiles["p50"],
        "p95": quantiles["p95"],
    }
    return {name: available[name] for name in REQUESTED_STATISTICS}


def _branch_payload(state, batch_result=None, is_final=False):
    stats_state = _stats_from_state(state) if is_final else state["stats_state"]
    candidate_results = int(state["candidate_results"])
    active_sketch = (
        state["canonical_sketch"]
        if (is_final or state["progress_sketch"] is None)
        else state["progress_sketch"]
    )
    if is_final and state["valid_results"] > 0:
        counts, edges, _, _ = state["canonical_sketch"].build_histogram(
            override_min=stats_state["min"],
            override_max=stats_state["max"],
        )
    else:
        counts = state["histogram_counts"]
        edges = state["histogram_edges"]
    payload = {
        "id": state["id"],
        "label": state["label"],
        "candidate_results": candidate_results,
        "valid_results": int(state["valid_results"]),
        "discarded_results": int(state["discarded_results"]),
        "stats": _stats_payload(
            stats_state,
            candidate_results,
            active_sketch,
            compute_quantiles=is_final or (MEMORY_MODE == "retain_all" and SEED_STRATEGY == "batch_index") or TOTAL_ITERATIONS <= EXACT_SAMPLE_RETENTION_LIMIT,
        ),
        "diagnostics": _numerical_diagnostics(stats_state, candidate_results),
        "histogram": _histogram_payload(
            counts,
            edges,
            stats_state["min"],
            stats_state["max"],
        ),
    }
    if batch_result is not None:
        payload["batch"] = {
            "candidate_results": batch_result["candidate_results"],
            "valid_results": batch_result["valid_results"],
            "discarded_results": batch_result["discarded_results"],
            "stats": _batch_item_stats(batch_result),
        }
    return payload


def _paired_payload(state, batch_result=None, is_final=False):
    if state is None:
        return None
    stats_state = _stats_from_state(state) if is_final else state["stats_state"]
    candidate_results = int(state["candidate_results"])
    active_sketch = (
        state["canonical_sketch"]
        if (is_final or state["progress_sketch"] is None)
        else state["progress_sketch"]
    )
    corr_mean = _canonical_correlation_mean(state) if is_final else _progress_correlation_mean(state)
    if is_final and state["valid_results"] > 0:
        histogram_counts, histogram_edges, _, _ = state["canonical_sketch"].build_histogram(
            override_min=stats_state["min"],
            override_max=stats_state["max"],
        )
    else:
        histogram_counts = state["histogram_counts"]
        histogram_edges = state["histogram_edges"]
    payload = {
        "id": state["id"],
        "label": state["label"],
        "candidate_results": candidate_results,
        "valid_results": int(state["valid_results"]),
        "discarded_results": int(state["discarded_results"]),
        "stats": _stats_payload(
            stats_state,
            candidate_results,
            active_sketch,
            compute_quantiles=is_final or (MEMORY_MODE == "retain_all" and SEED_STRATEGY == "batch_index") or TOTAL_ITERATIONS <= EXACT_SAMPLE_RETENTION_LIMIT,
        ),
        "diagnostics": _numerical_diagnostics(stats_state, candidate_results),
        "correlation_plus_minus_mean": corr_mean,
        "histogram": _histogram_payload(
            histogram_counts,
            histogram_edges,
            stats_state["min"],
            stats_state["max"],
        ),
    }
    if batch_result is not None:
        payload["batch"] = {
            "candidate_results": batch_result["candidate_results"],
            "valid_results": batch_result["valid_results"],
            "discarded_results": batch_result["discarded_results"],
            "stats": _batch_item_stats(batch_result),
            "correlation_plus_minus": batch_result["correlation_plus_minus"],
        }
    return payload


def _nested_array_payload_bytes(value, seen=None):
    """Count unique NumPy array payload bytes in a nested task/result structure."""
    if seen is None:
        seen = set()
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_nested_array_payload_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_nested_array_payload_bytes(item, seen) for item in value)
    return 0


def _retained_coordinator_payload_bytes(
    branch_states,
    paired_state,
    pending_logical_blocks=None,
    current_result=None,
    buffered_results=None,
):
    """Estimate retained coordinator arrays, bounded reorder buffers and active task payloads."""
    total_bytes = 0
    for state in branch_states.values():
        total_bytes += int(state["canonical_sketch"].retained_bytes())
        if state["progress_sketch"] is not None:
            total_bytes += int(state["progress_sketch"].retained_bytes())
        total_bytes += len(state["batch_stats"]) * 64
    if paired_state is not None:
        total_bytes += int(paired_state["canonical_sketch"].retained_bytes())
        if paired_state["progress_sketch"] is not None:
            total_bytes += int(paired_state["progress_sketch"].retained_bytes())
        total_bytes += len(paired_state["batch_stats"]) * 64
        total_bytes += len(paired_state["correlations"]) * 32
    if pending_logical_blocks:
        for lb_data in pending_logical_blocks.values():
            for lb_b in lb_data["branches"]:
                if lb_b["summary"] is not None:
                    total_bytes += int(lb_b["summary"]["centers"].nbytes + lb_b["summary"]["weights"].nbytes)
                elif lb_b["values"] is not None:
                    total_bytes += int(lb_b["values"].nbytes)
            if lb_data.get("paired") is not None:
                lb_p = lb_data["paired"]
                if lb_p["summary"] is not None:
                    total_bytes += int(lb_p["summary"]["centers"].nbytes + lb_p["summary"]["weights"].nbytes)
                elif lb_p["values"] is not None:
                    total_bytes += int(lb_p["values"].nbytes)
    if current_result is not None:
        total_bytes += _nested_array_payload_bytes(current_result)
    if buffered_results:
        total_bytes += _nested_array_payload_bytes(buffered_results)
    working_block = EFF_LOGICAL_BLOCK_SIZE if MEMORY_MODE == "bounded" else BATCH_SIZE
    total_bytes += int(working_block * ESTIMATED_BYTES_PER_SAMPLE)
    return total_bytes


def _metadata(worker_count, batch_count, runtime_extra=None):
    if SEED_STRATEGY == "philox_logical_block":
        rng_name = np.random.default_rng(np.random.Philox(np.random.SeedSequence([MASTER_SEED, 0, 0]))).bit_generator.__class__.__name__
        seed_desc = "numpy.SeedSequence([MASTER_SEED, logical_block_index, var_index]) -> Philox"
    else:
        rng_name = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 0])).bit_generator.__class__.__name__
        seed_desc = "numpy.SeedSequence([MASTER_SEED, batch_index])"
    meta = {
        "statistical_core_version": STATISTICAL_CORE_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "rng_bit_generator": rng_name,
        "seed_strategy": seed_desc,
        "scheduler_mode": SCHEDULER_MODE,
        "logical_block_size": int(EFF_LOGICAL_BLOCK_SIZE),
        "logical_block_count": int(LOGICAL_BLOCK_COUNT),
        "memory_mode": MEMORY_MODE,
        "optimizer_enabled": OPTIMIZER_ENABLED,
        "optimization_metrics": OPTIMIZATION_METRICS,
        "ast_cost_features": AST_COST_FEATURES,
        "cost_model_weights_seconds_per_sample": list(COST_MODEL_WEIGHTS),
        "static_per_sample_seconds": STATIC_PER_SAMPLE_SECONDS,
        "nonfinite_policy": "discard_and_report",
        "ci95_mean_type": "normal_approximation",
        "batch_count": int(batch_count),
        "worker_count": int(worker_count),
        "rss_measurement_method": (
            "resource.getrusage.ru_maxrss"
            if resource is not None
            else ("GetProcessMemoryInfo.PeakWorkingSetSize" if os.name == "nt" else "unavailable")
        ),
        "rss_sum_semantics": "sum_of_per_process_high_water_marks_not_simultaneous_aggregate",
        "retained_payload_semantics": (
            "estimated_unique_numpy_payload_plus_bounded_reorder_buffers_and_working_block"
        ),
        "max_global_sketch_points": int(MAX_GLOBAL_SKETCH_POINTS),
        "max_pending_logical_blocks": int(MAX_PENDING_LOGICAL_BLOCKS),
        "max_in_flight_tasks_per_worker": int(MAX_IN_FLIGHT_TASKS_PER_WORKER),
    }
    if runtime_extra:
        meta.update(runtime_extra)
    return meta


def _ingest_canonical_logical_block(lb_index, lb_data, branch_states, paired_state):
    for lb_branch in lb_data["branches"]:
        b_state = branch_states[lb_branch["id"]]
        if (
            MEMORY_MODE == "bounded"
            and lb_index > b_state["next_canonical_block"]
            and lb_index not in b_state["batch_stats"]
            and len(b_state["batch_stats"]) >= MAX_PENDING_LOGICAL_BLOCKS
        ):
            raise RuntimeError("Bounded coordinator exceeded the logical-block reorder buffer.")
        b_state["batch_stats"][lb_index] = lb_branch["stats_state"]
        if lb_branch["valid_results"] > 0:
            if lb_branch["summary"] is not None:
                b_state["canonical_sketch"].add_summary(lb_index, lb_branch["summary"])
            elif lb_branch["values"] is not None:
                b_state["canonical_sketch"].add_block(
                    lb_index,
                    lb_branch["values"],
                    precomputed_min=lb_branch["min"],
                    precomputed_max=lb_branch["max"],
                )

    if paired_state is not None and lb_data["paired"] is not None:
        lb_paired = lb_data["paired"]
        if (
            MEMORY_MODE == "bounded"
            and lb_index > paired_state["next_canonical_block"]
            and lb_index not in paired_state["batch_stats"]
            and len(paired_state["batch_stats"]) >= MAX_PENDING_LOGICAL_BLOCKS
        ):
            raise RuntimeError("Bounded paired state exceeded the logical-block reorder buffer.")
        if (
            MEMORY_MODE == "bounded"
            and lb_index > paired_state["next_corr_block"]
            and lb_index not in paired_state["correlations"]
            and len(paired_state["correlations"]) >= MAX_PENDING_LOGICAL_BLOCKS
        ):
            raise RuntimeError("Bounded paired correlation state exceeded its reorder buffer.")
        paired_state["batch_stats"][lb_index] = lb_paired["stats_state"]
        paired_state["correlations"][lb_index] = lb_paired["correlation_plus_minus"]
        if lb_paired["valid_results"] > 0:
            if lb_paired["summary"] is not None:
                paired_state["canonical_sketch"].add_summary(lb_index, lb_paired["summary"])
            elif lb_paired["values"] is not None:
                paired_state["canonical_sketch"].add_block(
                    lb_index,
                    lb_paired["values"],
                    precomputed_min=lb_paired["stats_state"]["min"],
                    precomputed_max=lb_paired["stats_state"]["max"],
                )


def _ingest_task_result(result, branch_states, paired_state, pending_logical_blocks, block_covered_samples):
    is_aligned = bool(result.get("aligned", True))

    # 1. Acumulador de progreso: incorpora unicamente los resultados de la tarea fisica completada
    for branch_result in result["branches"]:
        state = branch_states[branch_result["id"]]
        state["candidate_results"] += branch_result["candidate_results"]
        state["valid_results"] += branch_result["valid_results"]
        state["discarded_results"] += branch_result["discarded_results"]
        state["stats_state"] = _merge_stats(state["stats_state"], branch_result["stats_state"])
        if not is_aligned and branch_result["valid_results"] > 0:
            if state["progress_sketch"] is None:
                state["progress_sketch"] = _make_streaming_histogram()
            step = state["progress_step"]
            if branch_result["summary"] is not None:
                state["progress_sketch"].add_summary(step, branch_result["summary"])
            elif branch_result["values"] is not None:
                state["progress_sketch"].add_block(
                    step,
                    branch_result["values"],
                    precomputed_min=branch_result["min"],
                    precomputed_max=branch_result["max"],
                )
            state["progress_step"] = step + 1

    if paired_state is not None and result["paired"] is not None:
        paired = result["paired"]
        paired_state["candidate_results"] += paired["candidate_results"]
        paired_state["valid_results"] += paired["valid_results"]
        paired_state["discarded_results"] += paired["discarded_results"]
        paired_state["stats_state"] = _merge_stats(paired_state["stats_state"], paired["stats_state"])
        if paired["correlation_plus_minus"] is not None:
            paired_state["progress_corr_sum"], paired_state["progress_corr_comp"] = _neumaier_add(
                paired_state["progress_corr_sum"],
                paired_state["progress_corr_comp"],
                paired["correlation_plus_minus"],
            )
            paired_state["progress_corr_count"] += 1
        if not is_aligned and paired["valid_results"] > 0:
            if paired_state["progress_sketch"] is None:
                paired_state["progress_sketch"] = _make_streaming_histogram()
            p_step = paired_state["progress_step"]
            if paired["summary"] is not None:
                paired_state["progress_sketch"].add_summary(p_step, paired["summary"])
            elif paired["values"] is not None:
                paired_state["progress_sketch"].add_block(
                    p_step,
                    paired["values"],
                    precomputed_min=paired["stats_state"]["min"],
                    precomputed_max=paired["stats_state"]["max"],
                )
            paired_state["progress_step"] = p_step + 1

    # 2. Acumulador canonico: incorpora bloques logicos cuando su cobertura esta completa
    new_lb_indices = sorted(result["logical_blocks"])
    if SEED_STRATEGY == "batch_index" or is_aligned:
        # A single physical task may contain more logical blocks than the
        # coordinator's bounded reorder capacity (e.g. 125 blocks in a 5M
        # serial task). Those blocks are already ordered and contiguous, so
        # advance the canonical frontiers while ingesting that task instead
        # of temporarily retaining the whole task's summaries in batch_stats.
        # Keep the historical fast path unchanged for tasks that fit the cap.
        expected_first = next(iter(branch_states.values()))["next_canonical_block"] if branch_states else 0
        contiguous_from_frontier = bool(new_lb_indices) and new_lb_indices[0] == expected_first and all(
            lb_index == expected_first + offset
            for offset, lb_index in enumerate(new_lb_indices)
        )
        stream_contiguous_task = (
            MEMORY_MODE == "bounded"
            and len(new_lb_indices) > MAX_PENDING_LOGICAL_BLOCKS
            and contiguous_from_frontier
        )
        if stream_contiguous_task:
            # A single physical task may contain more logical blocks than the
            # coordinator's bounded reorder capacity (e.g. 125 blocks in a 5M
            # serial task). Those blocks are already contiguous from the
            # canonical frontier. Drain each one immediately instead of
            # temporarily retaining the whole task's summaries in batch_stats.
            for lb_index in new_lb_indices:
                _ingest_canonical_logical_block(
                    int(lb_index),
                    result["logical_blocks"][lb_index],
                    branch_states,
                    paired_state,
                )
                for b_state in branch_states.values():
                    _advance_state_frontier(b_state)
                if paired_state is not None:
                    _advance_paired_frontier(paired_state)
        else:
            # Preserve the original fast path for ordinary-sized tasks and
            # retain the explicit cap for genuinely out-of-order arrivals.
            for lb_index in new_lb_indices:
                _ingest_canonical_logical_block(
                    int(lb_index),
                    result["logical_blocks"][lb_index],
                    branch_states,
                    paired_state,
                )
    else:
        for lb_index, lb_data in result["logical_blocks"].items():
            if (
                MEMORY_MODE == "bounded"
                and int(lb_index) not in pending_logical_blocks
                and len(pending_logical_blocks) >= MAX_PENDING_LOGICAL_BLOCKS
            ):
                raise RuntimeError("Bounded coordinator exceeded the pending logical-block buffer.")
            pending_logical_blocks[int(lb_index)] = lb_data
        start_s = int(result["start_sample"])
        end_s = int(result["end_sample"])
        first_b = start_s // EFF_LOGICAL_BLOCK_SIZE
        last_b = (end_s - 1) // EFF_LOGICAL_BLOCK_SIZE
        for b in range(first_b, last_b + 1):
            b_start = b * EFF_LOGICAL_BLOCK_SIZE
            b_end = min(TOTAL_ITERATIONS, (b + 1) * EFF_LOGICAL_BLOCK_SIZE)
            overlap = max(0, min(end_s, b_end) - max(start_s, b_start))
            if (
                MEMORY_MODE == "bounded"
                and b not in block_covered_samples
                and len(block_covered_samples) >= MAX_PENDING_LOGICAL_BLOCKS
            ):
                raise RuntimeError("Bounded coordinator exceeded the block-coverage reorder buffer.")
            block_covered_samples[b] = block_covered_samples.get(b, 0) + overlap

        for lb_index in sorted(list(pending_logical_blocks.keys())):
            b_start = lb_index * EFF_LOGICAL_BLOCK_SIZE
            b_end = min(TOTAL_ITERATIONS, (lb_index + 1) * EFF_LOGICAL_BLOCK_SIZE)
            b_iters = b_end - b_start
            if block_covered_samples.get(lb_index, 0) >= b_iters:
                lb_data = pending_logical_blocks.pop(lb_index)
                block_covered_samples.pop(lb_index, None)
                _ingest_canonical_logical_block(lb_index, lb_data, branch_states, paired_state)

    for b_state in branch_states.values():
        _advance_state_frontier(b_state)
    if paired_state is not None:
        _advance_paired_frontier(paired_state)

    # 3. Actualizacion incremental del histograma de progreso (garantiza total_count == valid_results)
    for branch_index, branch_result in enumerate(result["branches"]):
        state = branch_states[branch_result["id"]]
        if state["valid_results"] <= 0:
            continue
        data_min = state["stats_state"]["min"]
        data_max = state["stats_state"]["max"]
        if not is_aligned or state["progress_sketch"] is not None:
            active_sketch = state["progress_sketch"] or state["canonical_sketch"]
            state["histogram_counts"], state["histogram_edges"], _, _ = active_sketch.build_histogram(
                override_min=data_min,
                override_max=data_max,
            )
            continue

        edges = state["histogram_edges"]
        range_changed = (
            edges is None
            or branch_result["min"] is None
            or float(edges[0]) > data_min
            or float(edges[-1]) < data_max
            or (data_min != data_max and (float(edges[0]) != data_min or float(edges[-1]) != data_max))
        )
        if range_changed or not new_lb_indices:
            state["histogram_counts"], state["histogram_edges"], _, _ = state["canonical_sketch"].build_histogram(
                override_min=data_min,
                override_max=data_max,
            )
        else:
            for lb_index in new_lb_indices:
                lb_branch = result["logical_blocks"][lb_index]["branches"][branch_index]
                if lb_branch["valid_results"] <= 0:
                    continue
                if lb_branch["summary"] is not None:
                    c, _ = np.histogram(
                        lb_branch["summary"]["centers"],
                        bins=state["histogram_edges"],
                        weights=lb_branch["summary"]["weights"],
                    )
                else:
                    c, _ = np.histogram(lb_branch["values"], bins=state["histogram_edges"])
                state["histogram_counts"] += c.astype(np.int64, copy=False)

    if paired_state is not None and result["paired"] is not None and paired_state["valid_results"] > 0:
        paired_min = paired_state["stats_state"]["min"]
        paired_max = paired_state["stats_state"]["max"]
        if not is_aligned or paired_state["progress_sketch"] is not None:
            active_paired_sketch = paired_state["progress_sketch"] or paired_state["canonical_sketch"]
        else:
            active_paired_sketch = paired_state["canonical_sketch"]
        (
            paired_state["histogram_counts"],
            paired_state["histogram_edges"],
            _,
            _,
        ) = active_paired_sketch.build_histogram(
            override_min=paired_min,
            override_max=paired_max,
        )


def _emit_batch_event(result, completed_batches, total_batches, processed_iterations, branch_states, paired_state):
    batch_branch_lookup = {branch["id"]: branch for branch in result["branches"]}
    branches_payload = [
        _branch_payload(state, batch_branch_lookup[state["id"]], is_final=False)
        for state in branch_states.values()
    ]
    valid_total = sum(branch["valid_results"] for branch in branches_payload)
    discarded_total = sum(branch["discarded_results"] for branch in branches_payload)
    paired_batch = result["paired"] if result["paired"] is not None else None

    _emit({
        "type": "batch",
        "batch_index": result["batch_index"],
        "completed_batches": completed_batches,
        "batch_count": total_batches,
        "processed_iterations": processed_iterations,
        "total_iterations": TOTAL_ITERATIONS,
        "progress": processed_iterations / TOTAL_ITERATIONS,
        "candidate_results": processed_iterations * BRANCH_COUNT,
        "valid_results": valid_total,
        "discarded_results": discarded_total,
        "batch": {"iterations": result["iterations"]},
        "branches": branches_payload,
        "paired": _paired_payload(paired_state, paired_batch, is_final=False),
        "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
        "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
        "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
        "done": False,
    })


def _run():
    _t_sim_start = time.perf_counter()
    coordinator_pid = os.getpid()
    worker_rss_by_pid = {}
    branch_states = {
        branch_id: _new_branch_state(branch_id, label)
        for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
    }
    paired_state = _new_paired_state() if BRANCH_COUNT == 2 else None
    pending_logical_blocks = {}
    block_covered_samples = {}
    processed_iterations = 0
    completed_batches = 0
    peak_retained_coordinator_bytes = 0
    peak_pending_logical_blocks_observed = 0
    peak_block_coverage_entries_observed = 0
    peak_branch_batch_stats_observed = 0
    peak_paired_batch_stats_observed = 0
    peak_paired_correlations_observed = 0
    peak_futures_observed = 0
    peak_completed_results_observed = 0

    pilot_elapsed = None
    pilot_per_sample = None
    effective_per_sample = STATIC_PER_SAMPLE_SECONDS
    pilot_result = None

    use_fixed_partition = (
        SCHEDULER_MODE == "fixed"
        or SEED_STRATEGY == "batch_index"
        or BATCH_SIZE < LOGICAL_BLOCK_SIZE
    )

    if use_fixed_partition:
        if SCHEDULER_MODE == "fixed" and SEED_STRATEGY != "batch_index" and BATCH_SIZE >= LOGICAL_BLOCK_SIZE:
            remaining_batches = _iter_static_schedule_tasks()
            remaining_task_count = STATIC_TASK_COUNT
            worker_count = STATIC_EFFECTIVE_WORKERS
            use_pool = STATIC_USE_PROCESS_POOL
        else:
            remaining_batches = _fixed_batch_plan()
            remaining_task_count = _fixed_batch_count()
            worker_count = min(MAX_WORKERS, max(1, remaining_task_count))
            use_pool = (worker_count > 1 and TOTAL_ITERATIONS >= LOGICAL_BLOCK_SIZE) or (
                SCHEDULER_MODE == "fixed" and TOTAL_ITERATIONS >= LOGICAL_BLOCK_SIZE
            )
    elif SCHEDULER_MODE == "ast_only":
        remaining_batches = _iter_static_schedule_tasks()
        remaining_task_count = STATIC_TASK_COUNT
        worker_count = STATIC_EFFECTIVE_WORKERS
        use_pool = STATIC_USE_PROCESS_POOL
    else:
        pilot_size = EFF_LOGICAL_BLOCK_SIZE
        t0 = time.perf_counter()
        pilot_result = _simulate_batch(0, pilot_size, 0, pilot_size)
        pilot_elapsed = time.perf_counter() - t0
        (
            remaining_batches,
            remaining_task_count,
            worker_count,
            use_pool,
            pilot_per_sample,
            effective_per_sample,
        ) = _plan_after_pilot(pilot_size, pilot_elapsed)

    total_batches = int(remaining_task_count) + (1 if pilot_result is not None else 0)
    serial_bounded_canonical = (
        not use_pool
        and MEMORY_MODE == "bounded"
        and TOTAL_ITERATIONS > EXACT_SAMPLE_RETENTION_LIMIT
    )
    effective_submission_order = (
        "canonical" if serial_bounded_canonical else TASK_COMPLETION_ORDER
    )
    runtime_extra = {
        "pilot_elapsed_seconds": pilot_elapsed,
        "pilot_per_sample_seconds": pilot_per_sample,
        "effective_per_sample_seconds": effective_per_sample,
        "use_process_pool": bool(use_pool),
        "max_in_flight_tasks": int(worker_count * MAX_IN_FLIGHT_TASKS_PER_WORKER) if use_pool else 1,
        "task_submission_order": effective_submission_order,
        "task_result_ingestion_order": "canonical_submission_order" if use_pool or serial_bounded_canonical else TASK_COMPLETION_ORDER,
        "task_delay_first_seconds": float(TASK_DELAY_FIRST_SECONDS),
    }

    _emit({
        "type": "start",
        "protocol_version": PROTOCOL_VERSION,
        "statistical_core_version": STATISTICAL_CORE_VERSION,
        "seed": str(MASTER_SEED),
        "iterations": TOTAL_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "batch_count": total_batches,
        "workers": worker_count,
        "histogram_bins": HISTOGRAM_BINS,
        "requested_statistics": list(REQUESTED_STATISTICS),
        "branches": [
            {"id": branch_id, "label": label}
            for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
        ],
        "has_plus_minus": BRANCH_COUNT == 2,
        "metadata": _metadata(worker_count, total_batches, runtime_extra),
    })

    def _record_and_emit(result, buffered_results=None):
        nonlocal completed_batches, processed_iterations, peak_retained_coordinator_bytes
        nonlocal peak_pending_logical_blocks_observed, peak_block_coverage_entries_observed
        nonlocal peak_branch_batch_stats_observed, peak_paired_batch_stats_observed
        nonlocal peak_paired_correlations_observed
        completed_batches += 1
        processed_iterations += result["iterations"]
        w_pid = int(result.get("worker_pid", coordinator_pid))
        w_rss = int(result.get("worker_peak_rss_bytes", 0))
        if w_pid != coordinator_pid and w_rss > 0:
            worker_rss_by_pid[w_pid] = max(worker_rss_by_pid.get(w_pid, 0), w_rss)
        _ingest_task_result(
            result,
            branch_states,
            paired_state,
            pending_logical_blocks,
            block_covered_samples,
        )
        peak_pending_logical_blocks_observed = max(
            peak_pending_logical_blocks_observed, len(pending_logical_blocks)
        )
        peak_block_coverage_entries_observed = max(
            peak_block_coverage_entries_observed, len(block_covered_samples)
        )
        if branch_states:
            peak_branch_batch_stats_observed = max(
                peak_branch_batch_stats_observed,
                max(len(state["batch_stats"]) for state in branch_states.values()),
            )
        if paired_state is not None:
            peak_paired_batch_stats_observed = max(
                peak_paired_batch_stats_observed, len(paired_state["batch_stats"])
            )
            peak_paired_correlations_observed = max(
                peak_paired_correlations_observed, len(paired_state["correlations"])
            )
        cur_ret = _retained_coordinator_payload_bytes(
            branch_states,
            paired_state,
            pending_logical_blocks,
            current_result=result,
            buffered_results=buffered_results,
        )
        if cur_ret > peak_retained_coordinator_bytes:
            peak_retained_coordinator_bytes = cur_ret
        _emit_batch_event(
            result,
            completed_batches,
            total_batches,
            processed_iterations,
            branch_states,
            paired_state,
        )

    if pilot_result is not None:
        _record_and_emit(pilot_result)

    if remaining_task_count > 0:
        if use_pool:
            max_in_flight = max(
                1,
                int(worker_count) * MAX_IN_FLIGHT_TASKS_PER_WORKER,
            )
            submission_iterator = iter(
                _iter_tasks_in_submission_order(remaining_batches, max_in_flight)
            )
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {}
                completed_results = {}
                submitted_count = 0
                processed_count = 0
                next_expected_task_index = 1 if pilot_result is not None else 0
                while processed_count < remaining_task_count:
                    while (
                        submitted_count < remaining_task_count
                        and len(futures) + len(completed_results) < max_in_flight
                    ):
                        try:
                            task = next(submission_iterator)
                        except StopIteration as exc:
                            raise RuntimeError("Lazy task stream ended before its declared task_count.") from exc
                        task_index = int(task[0])
                        submitted_future = executor.submit(_simulate_batch, *task)
                        futures[submitted_future] = task_index
                        submitted_count += 1
                        peak_futures_observed = max(peak_futures_observed, len(futures))
                        del task
                        del submitted_future

                    if futures:
                        done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for completed_future in done_futures:
                            task_index = futures.pop(completed_future)
                            completed_results[task_index] = completed_future.result()
                        peak_completed_results_observed = max(
                            peak_completed_results_observed, len(completed_results)
                        )
                        done_futures.clear()
                        del done_futures
                        try:
                            del completed_future
                        except UnboundLocalError:
                            pass

                        finished_future_payloads = []
                        for pending_future in futures:
                            if pending_future.done():
                                try:
                                    finished_future_payloads.append(pending_future.result())
                                except Exception:
                                    pass
                        buffered_retained = _retained_coordinator_payload_bytes(
                            branch_states,
                            paired_state,
                            pending_logical_blocks,
                            buffered_results=(completed_results, finished_future_payloads),
                        )
                        peak_retained_coordinator_bytes = max(
                            peak_retained_coordinator_bytes,
                            buffered_retained,
                        )
                        finished_future_payloads.clear()
                        del finished_future_payloads

                    while next_expected_task_index in completed_results:
                        result = completed_results.pop(next_expected_task_index)
                        _record_and_emit(result, buffered_results=completed_results)
                        del result
                        next_expected_task_index += 1
                        processed_count += 1

                    if (
                        not futures
                        and processed_count < remaining_task_count
                        and submitted_count >= remaining_task_count
                        and next_expected_task_index not in completed_results
                    ):
                        raise RuntimeError("Process task reorder buffer lost the next canonical task.")
                futures.clear()
                completed_results.clear()
                del futures
                del completed_results
                del submission_iterator
                try:
                    del pending_future
                except UnboundLocalError:
                    pass
        else:
            if serial_bounded_canonical:
                serial_tasks = remaining_batches
            else:
                serial_window = max(1, int(worker_count) * MAX_IN_FLIGHT_TASKS_PER_WORKER)
                serial_tasks = _iter_tasks_in_submission_order(remaining_batches, serial_window)
            for task in serial_tasks:
                result = _simulate_batch(*task)
                _record_and_emit(result)
                del result

    for lb_index in sorted(list(pending_logical_blocks.keys())):
        lb_data = pending_logical_blocks.pop(lb_index)
        block_covered_samples.pop(lb_index, None)
        _ingest_canonical_logical_block(lb_index, lb_data, branch_states, paired_state)

    for b_state in branch_states.values():
        if b_state["progress_sketch"] is not None:
            b_state["progress_sketch"].clear()
            b_state["progress_sketch"] = None
        _advance_state_frontier(b_state)
    if paired_state is not None:
        if paired_state["progress_sketch"] is not None:
            paired_state["progress_sketch"].clear()
            paired_state["progress_sketch"] = None
        _advance_paired_frontier(paired_state)

    branches_payload = [_branch_payload(state, is_final=True) for state in branch_states.values()]
    valid_total = sum(branch["valid_results"] for branch in branches_payload)
    discarded_total = sum(branch["discarded_results"] for branch in branches_payload)
    if valid_total == 0:
        raise RuntimeError("La simulacion termino sin ningun resultado numerico finito.")

    final_retained_bytes = _retained_coordinator_payload_bytes(
        branch_states, paired_state, pending_logical_blocks
    )
    peak_retained_coordinator_bytes = max(peak_retained_coordinator_bytes, final_retained_bytes)
    coordinator_peak_rss = _process_peak_rss_bytes()
    workers_peak_rss = int(sum(worker_rss_by_pid.values()))
    sum_of_process_peak_rss = int(coordinator_peak_rss + workers_peak_rss)

    runtime_extra["retained_coordinator_payload_bytes"] = int(final_retained_bytes)
    runtime_extra["peak_retained_coordinator_bytes"] = int(peak_retained_coordinator_bytes)
    runtime_extra["peak_memory_bytes"] = int(peak_retained_coordinator_bytes)
    runtime_extra["coordinator_peak_rss_bytes"] = int(coordinator_peak_rss)
    runtime_extra["workers_peak_rss_bytes"] = int(workers_peak_rss)
    runtime_extra["sum_of_process_peak_rss_bytes"] = sum_of_process_peak_rss
    runtime_extra["aggregate_peak_rss_bytes"] = sum_of_process_peak_rss
    runtime_extra["rss_measurement_available"] = bool(coordinator_peak_rss > 0)
    runtime_extra["max_pending_logical_blocks"] = int(MAX_PENDING_LOGICAL_BLOCKS)
    runtime_extra["max_pending_logical_blocks_observed"] = int(peak_pending_logical_blocks_observed)
    runtime_extra["max_block_coverage_entries_observed"] = int(peak_block_coverage_entries_observed)
    runtime_extra["max_branch_batch_stats_observed"] = int(peak_branch_batch_stats_observed)
    runtime_extra["max_paired_batch_stats_observed"] = int(peak_paired_batch_stats_observed)
    runtime_extra["max_paired_correlations_observed"] = int(peak_paired_correlations_observed)
    runtime_extra["max_futures_observed"] = int(peak_futures_observed)
    runtime_extra["max_completed_results_observed"] = int(peak_completed_results_observed)
    runtime_extra["simulation_compute_seconds"] = time.perf_counter() - _t_sim_start

    _emit({
        "type": "complete",
        "processed_iterations": processed_iterations,
        "total_iterations": TOTAL_ITERATIONS,
        "progress": 1.0,
        "candidate_results": TOTAL_ITERATIONS * BRANCH_COUNT,
        "valid_results": valid_total,
        "discarded_results": discarded_total,
        "branches": branches_payload,
        "paired": _paired_payload(paired_state, is_final=True),
        "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
        "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
        "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
        "metadata": _metadata(worker_count, total_batches, runtime_extra),
        "done": True,
    })


def main():
    multiprocessing.freeze_support()
    try:
        _run()
    except Exception as exc:
        _emit({
            "type": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "done": True,
        })
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
        return (
            dedent(template)
            .replace("__SHARED_STATISTICS_SOURCE__", _SHARED_STATISTICS_SOURCE)
            .replace("__SHARED_SCHEDULER_SOURCE__", _SHARED_SCHEDULER_SOURCE)
            .replace("__MASTER_SEED__", str(self.seed))
            .replace("__TOTAL_ITERATIONS__", str(self.program.iterations))
            .replace("__BATCH_SIZE__", str(self.config.batch_size))
            .replace("__LOGICAL_BLOCK_SIZE__", str(self.config.logical_block_size))
            .replace("__MAX_WORKERS__", str(self.config.max_workers))
            .replace("__HISTOGRAM_BINS__", str(self.config.histogram_bins))
            .replace("__SCHEDULER_MODE__", repr(normalized_mode))
            .replace("__SEED_STRATEGY__", repr(self.config.seed_strategy))
            .replace("__MEMORY_MODE__", repr(self.config.memory_mode))
            .replace("__TASK_COMPLETION_ORDER__", repr(self.config.task_completion_order))
            .replace("__TASK_DELAY_FIRST_SECONDS__", repr(float(self.config.task_delay_first_seconds)))
            .replace("__OPTIMIZER_ENABLED__", repr(self.config.optimize))
            .replace("__OPTIMIZATION_METRICS__", repr(self.optimization_metrics.to_dict()))
            .replace("__AST_COST_FEATURES__", repr(self.cost_features.to_dict()))
            .replace("__COST_MODEL_WEIGHTS__", repr(self.cost_model.weights_seconds_per_sample))
            .replace("__PRIOR_COST_MODEL_WEIGHTS__", repr(self.cost_model.prior_weights_seconds_per_sample))
            .replace("__STATIC_PER_SAMPLE_SECONDS__", repr(self.static_per_sample_seconds))
            .replace("__SERIAL_TASK_OVERHEAD_SECONDS__", repr(self.cost_model.serial_task_overhead_seconds))
            .replace("__PARALLEL_POOL_SPAWN_SECONDS__", repr(self.cost_model.parallel_pool_spawn_seconds))
            .replace("__PARALLEL_TASK_IPC_SECONDS__", repr(self.cost_model.parallel_task_ipc_seconds))
            .replace("__PRIOR_EQUIVALENT_SAMPLES__", repr(self.cost_model.prior_equivalent_samples))
            .replace("__ESTIMATED_BYTES_PER_SAMPLE__", repr(self.cost_features.estimated_bytes_per_sample))
            .replace(
                "__STATIC_TASK_GENERATION__",
                repr({
                    "total_iterations": self.static_schedule_plan.total_iterations,
                    "logical_block_size": self.static_schedule_plan.logical_block_size,
                    "start_logical_block": 0,
                    "initial_task_index": 0,
                    "blocks_per_task": self.static_schedule_plan.blocks_per_task,
                    "effective_workers": self.static_schedule_plan.effective_workers,
                }),
            )
            .replace("__STATIC_TASK_COUNT__", repr(self.static_schedule_plan.task_count))
            .replace("__STATIC_EFFECTIVE_WORKERS__", repr(self.static_schedule_plan.effective_workers))
            .replace("__STATIC_USE_PROCESS_POOL__", repr(self.static_schedule_plan.use_process_pool))
            .replace("__REQUESTED_STATISTICS__", requested)
            .replace("__REQUIRES_QUANTILES__", requires_quantiles)
            .replace("__BRANCH_IDS__", repr(branch_ids))
            .replace("__BRANCH_LABELS__", repr(branch_labels))
            .replace("__DISTRIBUTION_LINES_PHILOX__", indent(distribution_lines_philox, "    "))
            .replace("__DISTRIBUTION_LINES_LEGACY__", indent(distribution_lines_legacy, "    "))
            .replace("__RESULT_LINES__", indent(result_lines, "        "))
        )

    def _distribution_lines(self, *, use_philox_per_var: bool = True) -> str:
        lines: list[str] = []
        for var_index, distribution in enumerate(self.program.ast.distributions):
            source_name = distribution.variable
            name = self.variable_names[source_name]
            rng_expr = (
                f"_variable_rng(logical_block_index, {var_index})"
                if use_philox_per_var
                else "rng"
            )
            if isinstance(distribution, NormalDistributionNode):
                mean = self._decimal(distribution.mean, distribution.span)
                stddev = self._decimal(distribution.stddev, distribution.span)
                lines.append(
                    f"{name} = {rng_expr}.normal(loc={mean}, scale={stddev}, size=iterations).astype(np.float64, copy=False)"
                )
                continue

            if isinstance(distribution, UniformDistributionNode):
                minimum_value = NumpyExpressionVisitor._float64(distribution.minimum, distribution.span)
                maximum_value = NumpyExpressionVisitor._float64(distribution.maximum, distribution.span)
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
                    f"{name} = {rng_expr}.uniform(low={minimum}, high={maximum}, size=iterations).astype(np.float64, copy=False)"
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
                    f"{name} = {rng_expr}.choice(np.array([{values}], dtype=np.float64), "
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
                    f"_raw_{index} = _coerce_result_array(({branch}), iterations)",
                    f"branch_raw_values.append(_raw_{index})",
                    f"_finite_{index} = _raw_{index}[np.isfinite(_raw_{index})]",
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
            Diagnostic(code=code, phase=DiagnosticPhase.GENERATION, message=message, span=span)
        )


def generate_python_script(
    program: ValidatedProgram,
    config: GeneratorConfig | None = None,
    cost_model: InterpretableASTCostModel | None = None,
) -> str:
    return PythonCodeGenerator(program, config, cost_model=cost_model).generate()


def generate_source(
    source: str,
    config: GeneratorConfig | None = None,
    cost_model: InterpretableASTCostModel | None = None,
) -> str:
    max_it = config.max_iterations if config is not None else 1_000_000
    return generate_python_script(
        analyze_source(source, max_iterations=max_it),
        config,
        cost_model=cost_model,
    )
