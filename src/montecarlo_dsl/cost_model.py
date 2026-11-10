from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

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
    UniformDistributionNode,
    VariableReferenceNode,
)
from .semantic import ValidatedProgram


_QUANTILE_STATS = {Statistic.P05, Statistic.P50, Statistic.P95}

PRIOR_WEIGHTS_SECONDS_PER_SAMPLE: tuple[float, float, float, float] = (
    6.5e-8,   # theta_0_prior: intercepto base por muestra (65 ns)
    2.8e-8,   # theta_dist_prior: coste por unidad de distribution_cost_index (28 ns)
    2.6e-8,   # theta_arith_prior: coste por unidad de arithmetic_cost_index (26 ns)
    2.2e-8,   # theta_trans_prior: coste por unidad de transcendental_cost_index (22 ns)
)

DEFAULT_CALIBRATED_WEIGHTS_SECONDS_PER_SAMPLE: tuple[float, float, float, float] = (
    1.2624926371329765e-07,  # theta_0 calibrado sobre los 8 modelos de entrenamiento (126.249 ns)
    7.282681154485559e-09,  # theta_dist calibrado (7.283 ns)
    6.007353201571716e-08,  # theta_arith calibrado (60.074 ns)
    6.867690673466813e-09,  # theta_trans calibrado (6.868 ns)
)


@dataclass(frozen=True, slots=True)
class ASTCostFeatures:
    """Caracteristicas estaticas del modelo DSL extraidas del AST validado (Componente A).

    Para garantizar identificabilidad estadistica al calibrar con un numero moderado
    de modelos de referencia, los 11 conteos primitivos del AST se proyectan sobre 4
    indices estructurales compuestos interpretables en ``compact_feature_vector()``:
    1. Intercepto base por muestra (gestion de bucles y filtrado de finitud).
    2. Indice de coste de muestreo estocastico (distribuciones y soporte discreto).
    3. Indice de intensidad aritmetica vectorizada y sobrecoste de rama pareada ``\\pm``.
    4. Indice de operaciones trascendentes/no lineales (``np.power``, ``np.sqrt``) y cuantiles.
    """

    num_random_variables: int
    num_normal_vars: int
    num_uniform_vars: int
    num_discrete_vars: int
    discrete_support_total: int
    num_add_sub: int
    num_mul: int
    num_div_frac: int
    num_power: int
    num_sqrt: int
    num_unary: int
    ast_node_count: int
    ast_depth: int
    branch_count: int
    requested_stats_count: int
    requires_quantiles: bool
    estimated_bytes_per_sample: int

    @property
    def distribution_cost_index(self) -> float:
        return float(
            1.0 * self.num_uniform_vars
            + 2.0 * self.num_normal_vars
            + 2.6 * self.num_discrete_vars
            + 0.35 * self.discrete_support_total
        )

    @property
    def arithmetic_cost_index(self) -> float:
        b = float(self.branch_count)
        paired_overhead = 2.8 if self.branch_count == 2 else 0.0
        return float(
            b
            * (
                0.35 * (self.num_add_sub + self.num_unary)
                + 0.40 * self.num_mul
                + 0.80 * self.num_div_frac
            )
            + paired_overhead
        )

    @property
    def transcendental_cost_index(self) -> float:
        b = float(self.branch_count)
        quantile_index = 0.60 if self.requires_quantiles else 0.0
        return float(
            b * (1.80 * self.num_power + 1.00 * self.num_sqrt) + quantile_index
        )

    @property
    def weighted_operation_score(self) -> float:
        """Indice escalar compuesto interpretable de complejidad computacional por muestra."""
        return float(
            1.0
            + self.distribution_cost_index
            + self.arithmetic_cost_index
            + self.transcendental_cost_index
        )

    def compact_feature_vector(self) -> np.ndarray:
        """Vector parsimonioso de 4 indices estructurales compuestos interpretables."""
        return np.array(
            [
                1.0,
                self.distribution_cost_index,
                self.arithmetic_cost_index,
                self.transcendental_cost_index,
            ],
            dtype=np.float64,
        )

    def feature_vector(self) -> np.ndarray:
        """Alias hacia el vector estructural identificable de 4 componentes."""
        return self.compact_feature_vector()

    def estimated_batch_memory_bytes(self, batch_size: int) -> int:
        """Memoria estimada de trabajo (bytes) para un lote fisico de tamano ``batch_size``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return int(batch_size) * self.estimated_bytes_per_sample

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_random_variables": self.num_random_variables,
            "num_normal_vars": self.num_normal_vars,
            "num_uniform_vars": self.num_uniform_vars,
            "num_discrete_vars": self.num_discrete_vars,
            "discrete_support_total": self.discrete_support_total,
            "num_add_sub": self.num_add_sub,
            "num_mul": self.num_mul,
            "num_div_frac": self.num_div_frac,
            "num_power": self.num_power,
            "num_sqrt": self.num_sqrt,
            "num_unary": self.num_unary,
            "ast_node_count": self.ast_node_count,
            "ast_depth": self.ast_depth,
            "branch_count": self.branch_count,
            "requested_stats_count": self.requested_stats_count,
            "requires_quantiles": self.requires_quantiles,
            "estimated_bytes_per_sample": self.estimated_bytes_per_sample,
            "distribution_cost_index": round(self.distribution_cost_index, 4),
            "arithmetic_cost_index": round(self.arithmetic_cost_index, 4),
            "transcendental_cost_index": round(self.transcendental_cost_index, 4),
            "weighted_operation_score": round(self.weighted_operation_score, 4),
        }


def extract_ast_cost_features(program: ValidatedProgram) -> ASTCostFeatures:
    """Calcula las caracteristicas estaticas de coste a partir de ``ValidatedProgram``."""
    num_normal = 0
    num_uniform = 0
    num_discrete = 0
    discrete_support = 0

    for dist in program.ast.distributions:
        if isinstance(dist, NormalDistributionNode):
            num_normal += 1
        elif isinstance(dist, UniformDistributionNode):
            num_uniform += 1
        elif isinstance(dist, DiscreteDistributionNode):
            num_discrete += 1
            discrete_support += len(dist.pairs)

    counts = {
        "add_sub": 0,
        "mul": 0,
        "div_frac": 0,
        "power": 0,
        "sqrt": 0,
        "unary": 0,
        "nodes": 0,
    }

    def _walk(node: ExpressionNode) -> int:
        counts["nodes"] += 1
        if isinstance(node, (NumberLiteralNode, VariableReferenceNode)):
            return 1
        if isinstance(node, UnaryOperationNode):
            counts["unary"] += 1
            return 1 + _walk(node.operand)
        if isinstance(node, BinaryOperationNode):
            if node.operator in (BinaryOperator.ADD, BinaryOperator.SUBTRACT):
                counts["add_sub"] += 1
            elif node.operator is BinaryOperator.MULTIPLY:
                counts["mul"] += 1
            elif node.operator is BinaryOperator.DIVIDE:
                counts["div_frac"] += 1
            left_d = _walk(node.left)
            right_d = _walk(node.right)
            return 1 + max(left_d, right_d)
        if isinstance(node, PlusMinusNode):
            counts["add_sub"] += 2
            left_d = _walk(node.left)
            right_d = _walk(node.right)
            return 1 + max(left_d, right_d)
        if isinstance(node, FractionNode):
            counts["div_frac"] += 1
            num_d = _walk(node.numerator)
            den_d = _walk(node.denominator)
            return 1 + max(num_d, den_d)
        if isinstance(node, PowerNode):
            counts["power"] += 1
            base_d = _walk(node.base)
            exp_d = _walk(node.exponent)
            return 1 + max(base_d, exp_d)
        if isinstance(node, SqrtNode):
            counts["sqrt"] += 1
            return 1 + _walk(node.expression)
        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    depth = _walk(program.ast.model.expression)
    branch_count = 2 if program.has_plus_minus else 1
    requires_quantiles = any(stat in _QUANTILE_STATS for stat in program.requested_statistics)

    active_arrays = (
        len(program.random_variables)
        + branch_count * 3
        + (2 if branch_count == 2 else 0)
    )
    estimated_bytes_per_sample = active_arrays * 8

    return ASTCostFeatures(
        num_random_variables=len(program.random_variables),
        num_normal_vars=num_normal,
        num_uniform_vars=num_uniform,
        num_discrete_vars=num_discrete,
        discrete_support_total=discrete_support,
        num_add_sub=counts["add_sub"],
        num_mul=counts["mul"],
        num_div_frac=counts["div_frac"],
        num_power=counts["power"],
        num_sqrt=counts["sqrt"],
        num_unary=counts["unary"],
        ast_node_count=counts["nodes"],
        ast_depth=depth,
        branch_count=branch_count,
        requested_stats_count=len(program.requested_statistics),
        requires_quantiles=requires_quantiles,
        estimated_bytes_per_sample=estimated_bytes_per_sample,
    )


@dataclass(frozen=True, slots=True)
class CalibratedCostEstimate:
    """Resultado de la estimacion de coste por muestra (prior estatico, piloto y contraccion ponderada)."""

    mode: str
    static_per_sample_seconds: float
    pilot_per_sample_seconds: float | None
    effective_per_sample_seconds: float
    pilot_weight_alpha: float
    pilot_iterations: int

    def estimate_work_seconds(self, iterations: int) -> float:
        return max(0.0, float(iterations) * self.effective_per_sample_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "static_per_sample_seconds": self.static_per_sample_seconds,
            "pilot_per_sample_seconds": self.pilot_per_sample_seconds,
            "effective_per_sample_seconds": self.effective_per_sample_seconds,
            "pilot_weight_alpha": round(self.pilot_weight_alpha, 4),
            "pilot_iterations": self.pilot_iterations,
        }


@dataclass(frozen=True, slots=True)
class InterpretableASTCostModel:
    """Modelo lineal interpretable de 4 indices estructurales compuestos con calibracion piloto.

    Los 4 coeficientes corresponden a ``[base, distribution_index, arithmetic_index, transcendental_index]``
    en segundos por muestra. Por defecto utiliza los pesos calibrados mediante regresion Ridge
    con restricciones de cota inferior (KKT exacta) sobre los 8 modelos de entrenamiento de referencia,
    y se combina en ejecucion con la medicion del bloque piloto mediante contraccion ponderada por
    tamano muestral (sample-size-weighted shrinkage).
    """

    weights_seconds_per_sample: tuple[float, float, float, float] = (
        DEFAULT_CALIBRATED_WEIGHTS_SECONDS_PER_SAMPLE
    )
    prior_weights_seconds_per_sample: tuple[float, float, float, float] = (
        PRIOR_WEIGHTS_SECONDS_PER_SAMPLE
    )
    serial_task_overhead_seconds: float = 0.00035
    parallel_pool_spawn_seconds: float = 0.18
    parallel_task_ipc_seconds: float = 0.006
    prior_equivalent_samples: int = 1_500

    @classmethod
    def uncalibrated_prior(cls) -> "InterpretableASTCostModel":
        """Devuelve una instancia con los pesos del prior estructural de referencia sin calibrar."""
        return cls(weights_seconds_per_sample=PRIOR_WEIGHTS_SECONDS_PER_SAMPLE)

    def estimate_static_per_sample_seconds(self, features: ASTCostFeatures) -> float:
        vec = features.compact_feature_vector()
        w = np.asarray(self.weights_seconds_per_sample, dtype=np.float64)
        estimate = float(np.dot(vec, w))
        return max(5.0e-9, estimate)

    def estimate_total_seconds(
        self,
        features: ASTCostFeatures,
        iterations: int,
        *,
        num_tasks: int = 1,
        workers: int = 1,
    ) -> float:
        per_sample = self.estimate_static_per_sample_seconds(features)
        compute = float(iterations) * per_sample
        if workers <= 1:
            return compute + float(num_tasks) * self.serial_task_overhead_seconds
        efficiency = max(0.55, 1.0 - 0.08 * (workers - 1))
        return (
            self.parallel_pool_spawn_seconds
            + compute / (workers * efficiency)
            + float(num_tasks) * self.parallel_task_ipc_seconds
        )

    def calibrate_with_pilot(
        self,
        features: ASTCostFeatures,
        *,
        pilot_iterations: int = 0,
        pilot_elapsed_seconds: float | None = None,
        mode: str = "hybrid",
    ) -> CalibratedCostEstimate:
        static_cost = self.estimate_static_per_sample_seconds(features)

        if mode == "ast_only" or pilot_elapsed_seconds is None or pilot_iterations <= 0:
            return CalibratedCostEstimate(
                mode=mode,
                static_per_sample_seconds=static_cost,
                pilot_per_sample_seconds=None,
                effective_per_sample_seconds=static_cost,
                pilot_weight_alpha=0.0,
                pilot_iterations=0,
            )

        # Descontamos el sobrecoste fijo de invocacion del bloque piloto
        block_call_overhead = self.serial_task_overhead_seconds * (1.6 if features.branch_count == 2 else 1.0)
        net_pilot = max(1.0e-6, float(pilot_elapsed_seconds) - block_call_overhead)
        pilot_cost = max(5.0e-9, net_pilot / float(pilot_iterations))

        if mode == "pilot_only":
            return CalibratedCostEstimate(
                mode="pilot_only",
                static_per_sample_seconds=static_cost,
                pilot_per_sample_seconds=pilot_cost,
                effective_per_sample_seconds=pilot_cost,
                pilot_weight_alpha=1.0,
                pilot_iterations=int(pilot_iterations),
            )

        # Contraccion lineal ponderada por tamano muestral (sample-size-weighted shrinkage)
        alpha = float(pilot_iterations) / float(pilot_iterations + self.prior_equivalent_samples)
        alpha = min(0.88, max(0.15, alpha))
        effective = (1.0 - alpha) * static_cost + alpha * pilot_cost
        return CalibratedCostEstimate(
            mode="hybrid",
            static_per_sample_seconds=static_cost,
            pilot_per_sample_seconds=pilot_cost,
            effective_per_sample_seconds=effective,
            pilot_weight_alpha=alpha,
            pilot_iterations=int(pilot_iterations),
        )

    @staticmethod
    def design_matrix_diagnostics(features_list: Sequence[ASTCostFeatures]) -> dict[str, float]:
        """Calcula rango, numero de condicion (bruto y estandarizado) y correlacion entre indices."""
        if not features_list:
            return {
                "matrix_rank": 0.0,
                "condition_number_raw": 0.0,
                "condition_number_standardized": 0.0,
                "arith_trans_correlation": 0.0,
            }
        X = np.vstack([f.compact_feature_vector() for f in features_list])
        rank = int(np.linalg.matrix_rank(X))
        cond_raw = float(np.linalg.cond(X))
        X_nonconst = X[:, 1:]
        stds = np.std(X_nonconst, axis=0)
        if np.all(stds > 0):
            X_std = (X_nonconst - np.mean(X_nonconst, axis=0)) / stds
            cond_std = float(np.linalg.cond(X_std))
            corr = np.corrcoef(X_nonconst.T)
            arith_trans_corr = float(corr[1, 2])
        else:
            cond_std = cond_raw
            arith_trans_corr = 0.0
        return {
            "matrix_rank": float(rank),
            "condition_number_raw": round(cond_raw, 4),
            "condition_number_standardized": round(cond_std, 4),
            "arith_trans_correlation": round(arith_trans_corr, 4),
        }

    @classmethod
    def fit_from_observations(
        cls,
        observations: Sequence[tuple[ASTCostFeatures, int, float]],
        *,
        ridge_lambda: float = 1e-2,
        lower_bound_ratio: float = 0.25,
    ) -> "InterpretableASTCostModel":
        """Resuelve de forma exacta (conjunto activo KKT) la regresion Ridge con cotas inferiores:

        min_{theta >= lower_bound_ratio * theta_prior}
            ||X * theta - y||_2^2 + lambda_eff * ||(theta - theta_prior) / theta_prior||_2^2
        donde lambda_eff = ridge_lambda * tr(X_scaled^T X_scaled) / p mantiene consistencia dimensional.
        """
        default_model = cls()
        if not observations:
            return default_model

        prior_w = np.asarray(default_model.prior_weights_seconds_per_sample, dtype=np.float64)
        rows = []
        targets = []
        for features, iterations, elapsed_seconds in observations:
            if iterations <= 0:
                continue
            num_blocks = max(1, iterations // 5_000)
            overhead = num_blocks * default_model.serial_task_overhead_seconds * 0.25
            net = max(1.0e-6, float(elapsed_seconds) - overhead)
            per_sample = net / float(iterations)
            rows.append(features.compact_feature_vector())
            targets.append(per_sample)

        if not rows:
            return default_model

        X = np.vstack(rows)
        y = np.asarray(targets, dtype=np.float64)
        p = int(X.shape[1])

        scale = np.maximum(prior_w, 1.0e-9)
        X_scaled = X * scale
        y_residual = y - X @ prior_w
        gram_data = X_scaled.T @ X_scaled
        trace_scale = max(1.0e-30, float(np.trace(gram_data)) / float(p))
        lambda_eff = float(ridge_lambda) * trace_scale

        H = gram_data + lambda_eff * np.eye(p, dtype=np.float64)
        g = X_scaled.T @ y_residual
        lb_delta = np.full(p, float(lower_bound_ratio) - 1.0, dtype=np.float64)

        # Enumeracion exacta de conjuntos activos (2^4 = 16 caras) para el programa cuadratico
        # estrictamente convexo sobre d = (theta - theta_prior) / theta_prior >= lb_delta.
        best_obj = float("inf")
        best_delta = np.zeros(p, dtype=np.float64)

        for mask in range(1 << p):
            active = [i for i in range(p) if (mask >> i) & 1]
            free = [i for i in range(p) if not ((mask >> i) & 1)]
            d = np.empty(p, dtype=np.float64)
            d[active] = lb_delta[active]
            if free:
                H_ff = H[np.ix_(free, free)]
                rhs_f = g[free] - H[np.ix_(free, active)] @ d[active] if active else g[free]
                d[free] = np.linalg.solve(H_ff, rhs_f)
            # Factibilidad primal en variables libres
            if np.all(d >= lb_delta - 1.0e-12):
                d = np.maximum(d, lb_delta)
                resid = X_scaled @ d - y_residual
                obj = float(resid @ resid + lambda_eff * (d @ d))
                if obj < best_obj:
                    best_obj = obj
                    best_delta = d

        fitted_w = prior_w + best_delta * scale
        return cls(
            weights_seconds_per_sample=(
                float(fitted_w[0]),
                float(fitted_w[1]),
                float(fitted_w[2]),
                float(fitted_w[3]),
            ),
            prior_weights_seconds_per_sample=default_model.prior_weights_seconds_per_sample,
            serial_task_overhead_seconds=default_model.serial_task_overhead_seconds,
            parallel_pool_spawn_seconds=default_model.parallel_pool_spawn_seconds,
            parallel_task_ipc_seconds=default_model.parallel_task_ipc_seconds,
            prior_equivalent_samples=default_model.prior_equivalent_samples,
        )
