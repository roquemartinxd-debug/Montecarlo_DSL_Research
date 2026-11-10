"""Compilador para el DSL de simulacion Monte Carlo."""

__version__ = "1.3.1"

from .compiler import CompiledSimulation, compile_program, compile_source
from .cost_model import (
    ASTCostFeatures,
    CalibratedCostEstimate,
    InterpretableASTCostModel,
    extract_ast_cost_features,
)
from .diagnostics import CompilationError, Diagnostic, DiagnosticPhase
from .generator import (
    ExpressionCode,
    GeneratorConfig,
    NumpyExpressionVisitor,
    PythonCodeGenerator,
    generate_python_script,
    generate_source,
)
from .lexer import Lexer, tokenize
from .optimizer import ASTOptimizer, OptimizationMetrics, count_ast_nodes, optimize_program
from .parser import Parser, parse, parse_tokens
from .scheduler import (
    AdaptiveMonteCarloScheduler,
    ExecutionSchedulePlan,
    PhysicalTaskSpec,
)
from .semantic import SemanticAnalyzer, ValidatedProgram, analyze, analyze_source
from .source import SourcePosition, SourceSpan
from .statistics import (
    BlockHistogramSummary,
    BoundedStreamingHistogram,
    RunningStats,
    compact_weighted_sketch,
    merge_many,
    weighted_quantiles,
    wilson_interval,
)
from .symbols import DistributionKind, Symbol, SymbolKind, SymbolTable
from .tokens import Token, TokenType

__all__ = [
    "__version__",
    "ASTCostFeatures",
    "ASTOptimizer",
    "AdaptiveMonteCarloScheduler",
    "BlockHistogramSummary",
    "BoundedStreamingHistogram",
    "CalibratedCostEstimate",
    "CompilationError",
    "CompiledSimulation",
    "Diagnostic",
    "DiagnosticPhase",
    "DistributionKind",
    "ExecutionSchedulePlan",
    "ExpressionCode",
    "GeneratorConfig",
    "InterpretableASTCostModel",
    "Lexer",
    "NumpyExpressionVisitor",
    "OptimizationMetrics",
    "Parser",
    "PhysicalTaskSpec",
    "PythonCodeGenerator",
    "RunningStats",
    "SemanticAnalyzer",
    "SourcePosition",
    "SourceSpan",
    "Symbol",
    "SymbolKind",
    "SymbolTable",
    "Token",
    "TokenType",
    "ValidatedProgram",
    "analyze",
    "analyze_source",
    "compact_weighted_sketch",
    "compile_program",
    "compile_source",
    "count_ast_nodes",
    "extract_ast_cost_features",
    "generate_python_script",
    "generate_source",
    "merge_many",
    "optimize_program",
    "parse",
    "parse_tokens",
    "tokenize",
    "weighted_quantiles",
    "wilson_interval",
]
