"""Compilador para el DSL de simulacion Monte Carlo."""

__version__ = "1.2.0"

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
from .parser import Parser, parse, parse_tokens
from .semantic import SemanticAnalyzer, ValidatedProgram, analyze, analyze_source
from .source import SourcePosition, SourceSpan
from .statistics import RunningStats, merge_many, wilson_interval
from .symbols import DistributionKind, Symbol, SymbolKind, SymbolTable
from .tokens import Token, TokenType

__all__ = [
    "__version__",
    "CompilationError",
    "Diagnostic",
    "DiagnosticPhase",
    "DistributionKind",
    "ExpressionCode",
    "GeneratorConfig",
    "Lexer",
    "NumpyExpressionVisitor",
    "Parser",
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
    "generate_python_script",
    "generate_source",
    "merge_many",
    "parse",
    "parse_tokens",
    "tokenize",
    "wilson_interval",
]
