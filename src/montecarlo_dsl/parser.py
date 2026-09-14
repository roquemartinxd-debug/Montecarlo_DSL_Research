from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Iterable

from .ast import (
    BinaryOperationNode,
    BinaryOperator,
    DiscreteDistributionNode,
    DistributionNode,
    ExpressionNode,
    FractionNode,
    IterationsNode,
    ModelDeclarationNode,
    NormalDistributionNode,
    NumberLiteralNode,
    PlusMinusNode,
    PowerNode,
    ProbabilityPairNode,
    ProgramNode,
    ResponseNode,
    SqrtNode,
    Statistic,
    UnaryOperationNode,
    UnaryOperator,
    UniformDistributionNode,
    VariableReferenceNode,
)
from .diagnostics import CompilationError, Diagnostic, DiagnosticPhase
from .lexer import tokenize
from .source import SourceSpan
from .tokens import Token, TokenType


_CONFIGURATION_STARTERS = {
    TokenType.NORMAL,
    TokenType.DISTRIB,
    TokenType.UNIFORM,
}

_IMPLICIT_FACTOR_STARTERS = {
    TokenType.VARIABLE,
    TokenType.FRAC,
    TokenType.SQRT,
    TokenType.LBRACE,
}

_STATISTICS = {
    TokenType.AVG: Statistic.AVG,
    TokenType.MIN: Statistic.MIN,
    TokenType.MAX: Statistic.MAX,
    TokenType.VAR: Statistic.VAR,
    TokenType.STD: Statistic.STD,
    TokenType.COUNT: Statistic.COUNT,
    TokenType.VALID_RATE: Statistic.VALID_RATE,
    TokenType.DISCARD_RATE: Statistic.DISCARD_RATE,
    TokenType.P05: Statistic.P05,
    TokenType.P50: Statistic.P50,
    TokenType.P95: Statistic.P95,
}


class Parser:
    """Parser descendente recursivo para la BNF definitiva del DSL."""

    def __init__(self, tokens: Iterable[Token]):
        self.tokens = tuple(tokens)
        if not self.tokens:
            raise ValueError("Parser requires at least the EOF token")
        if self.tokens[-1].type is not TokenType.EOF:
            raise ValueError("Token stream must end with EOF")
        self._index = 0

    def parse(self) -> ProgramNode:
        start = self._current().span.start

        model = self._parse_model_declaration()
        distributions = self._parse_configurations()
        iterations = self._parse_iterations()
        response = self._parse_response()
        eof = self._consume(TokenType.EOF, "Se esperaba el fin de la entrada.")

        return ProgramNode(
            span=SourceSpan(start, eof.span.end),
            model=model,
            distributions=tuple(distributions),
            iterations=iterations,
            response=response,
        )

    def _parse_model_declaration(self) -> ModelDeclarationNode:
        model_token = self._consume(TokenType.MODEL, "El programa debe iniciar con \\model.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' despues de \\model.")
        output = self._consume(TokenType.VARIABLE, "Se esperaba la variable resultado del modelo.")
        self._consume(TokenType.ASSIGN, "Se esperaba '=' despues de la variable resultado.")
        expression = self._parse_expression()
        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar \\model.")

        return ModelDeclarationNode(
            span=SourceSpan(model_token.span.start, closing.span.end),
            output_name=str(output.value),
            expression=expression,
        )

    def _parse_configurations(self) -> list[DistributionNode]:
        if self._current().type not in _CONFIGURATION_STARTERS:
            self._error(
                code="SYN010",
                message="Se esperaba al menos una configuracion de distribucion antes de \\iter.",
                expected=("NORMAL", "DISTRIB", "UNIFORM"),
            )

        distributions: list[DistributionNode] = []
        while self._current().type in _CONFIGURATION_STARTERS:
            distributions.append(self._parse_configuration())
        return distributions

    def _parse_configuration(self) -> DistributionNode:
        token_type = self._current().type
        if token_type is TokenType.NORMAL:
            return self._parse_normal_distribution()
        if token_type is TokenType.DISTRIB:
            return self._parse_discrete_distribution()
        if token_type is TokenType.UNIFORM:
            return self._parse_uniform_distribution()

        self._error(
            code="SYN011",
            message="Se esperaba una configuracion de distribucion.",
            expected=("NORMAL", "DISTRIB", "UNIFORM"),
        )

    def _parse_normal_distribution(self) -> NormalDistributionNode:
        start_token = self._consume(TokenType.NORMAL, "Se esperaba \\normal.")
        variable = self._consume(TokenType.VARIABLE, "Se esperaba una variable despues de \\normal.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' en la distribucion normal.")
        mean, _ = self._parse_signed_decimal()
        self._consume(TokenType.COMMA, "Se esperaba ',' entre media y desviacion estandar.")
        stddev, _ = self._parse_signed_decimal()
        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar \\normal.")

        return NormalDistributionNode(
            span=SourceSpan(start_token.span.start, closing.span.end),
            variable=str(variable.value),
            mean=mean,
            stddev=stddev,
        )

    def _parse_discrete_distribution(self) -> DiscreteDistributionNode:
        start_token = self._consume(TokenType.DISTRIB, "Se esperaba \\distrib.")
        variable = self._consume(TokenType.VARIABLE, "Se esperaba una variable despues de \\distrib.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' en la distribucion discreta.")

        if self._current().type is not TokenType.LBRACE:
            self._error(
                code="SYN020",
                message="Una distribucion discreta requiere al menos un par {valor,probabilidad}.",
                expected=("LBRACE",),
            )

        pairs: list[ProbabilityPairNode] = []
        pairs.append(self._parse_probability_pair())
        while self._match(TokenType.COMMA):
            if self._current().type is not TokenType.LBRACE:
                self._error(
                    code="SYN021",
                    message="Se esperaba otro par {valor,probabilidad} despues de ','.",
                    expected=("LBRACE",),
                )
            pairs.append(self._parse_probability_pair())

        if self._current().type is TokenType.LBRACE:
            self._error(
                code="SYN022",
                message=(
                    "Los pares de \\distrib deben separarse mediante comas: "
                    "{valor,probabilidad},{valor,probabilidad}."
                ),
                expected=("COMMA", "RBRACE"),
            )

        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar \\distrib.")
        return DiscreteDistributionNode(
            span=SourceSpan(start_token.span.start, closing.span.end),
            variable=str(variable.value),
            pairs=tuple(pairs),
        )

    def _parse_probability_pair(self) -> ProbabilityPairNode:
        opening = self._consume(TokenType.LBRACE, "Se esperaba '{' al iniciar el par de probabilidad.")
        value, _ = self._parse_signed_decimal()
        self._consume(TokenType.COMMA, "Se esperaba ',' entre el valor y su probabilidad.")
        probability = self._consume(TokenType.NUMBER, "La probabilidad debe ser un NUMBER sin signo.")
        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' al cerrar el par de probabilidad.")

        return ProbabilityPairNode(
            span=SourceSpan(opening.span.start, closing.span.end),
            value=value,
            probability=self._number_value(probability),
        )

    def _parse_uniform_distribution(self) -> UniformDistributionNode:
        start_token = self._consume(TokenType.UNIFORM, "Se esperaba \\uniform.")
        variable = self._consume(TokenType.VARIABLE, "Se esperaba una variable despues de \\uniform.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' en la distribucion uniforme.")
        minimum, _ = self._parse_signed_decimal()
        self._consume(TokenType.COMMA, "Se esperaba ',' entre minimo y maximo.")
        maximum, _ = self._parse_signed_decimal()
        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar \\uniform.")

        return UniformDistributionNode(
            span=SourceSpan(start_token.span.start, closing.span.end),
            variable=str(variable.value),
            minimum=minimum,
            maximum=maximum,
        )

    def _parse_iterations(self) -> IterationsNode:
        start_token = self._consume(TokenType.ITER, "Se esperaba \\iter despues de las distribuciones.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' despues de \\iter.")
        number = self._consume(TokenType.NUMBER, "Se esperaba un NUMBER como cantidad de iteraciones.")
        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar \\iter.")

        return IterationsNode(
            span=SourceSpan(start_token.span.start, closing.span.end),
            lexeme=number.lexeme,
            value=self._number_value(number),
        )

    def _parse_response(self) -> ResponseNode:
        start_token = self._consume(TokenType.RESPONSE, "Se esperaba \\response despues de \\iter.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' despues de \\response.")

        statistics = [self._parse_statistic()]
        while self._match(TokenType.COMMA):
            statistics.append(self._parse_statistic())

        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar \\response.")
        return ResponseNode(
            span=SourceSpan(start_token.span.start, closing.span.end),
            statistics=tuple(statistics),
        )

    def _parse_statistic(self) -> Statistic:
        current = self._current()
        statistic = _STATISTICS.get(current.type)
        if statistic is None:
            self._error(
                code="SYN030",
                message="Se esperaba un estadistico valido dentro de \response.",
                expected=tuple(token.name for token in _STATISTICS),
            )
        self._advance()
        return statistic

    def _parse_expression(self) -> ExpressionNode:
        return self._parse_sum()

    def _parse_sum(self) -> ExpressionNode:
        expression = self._parse_product()

        while self._current().type in {TokenType.PLUS, TokenType.MINUS, TokenType.PM}:
            operator = self._advance()
            right = self._parse_product()
            span = SourceSpan(expression.span.start, right.span.end)

            if operator.type is TokenType.PM:
                expression = PlusMinusNode(span=span, left=expression, right=right)
            else:
                expression = BinaryOperationNode(
                    span=span,
                    operator=(
                        BinaryOperator.ADD
                        if operator.type is TokenType.PLUS
                        else BinaryOperator.SUBTRACT
                    ),
                    left=expression,
                    right=right,
                    implicit=False,
                )

        return expression

    def _parse_product(self) -> ExpressionNode:
        expression = self._parse_unary()

        while True:
            current_type = self._current().type

            if current_type in {TokenType.MULT, TokenType.DIV}:
                operator = self._advance()
                right = self._parse_unary()
                expression = BinaryOperationNode(
                    span=SourceSpan(expression.span.start, right.span.end),
                    operator=(
                        BinaryOperator.MULTIPLY
                        if operator.type is TokenType.MULT
                        else BinaryOperator.DIVIDE
                    ),
                    left=expression,
                    right=right,
                    implicit=False,
                )
                continue

            if current_type in _IMPLICIT_FACTOR_STARTERS:
                right = self._parse_implicit_factor()
                expression = BinaryOperationNode(
                    span=SourceSpan(expression.span.start, right.span.end),
                    operator=BinaryOperator.MULTIPLY,
                    left=expression,
                    right=right,
                    implicit=True,
                )
                continue

            return expression

    def _parse_unary(self) -> ExpressionNode:
        if self._current().type in {TokenType.PLUS, TokenType.MINUS}:
            operator = self._advance()
            operand = self._parse_power()
            return UnaryOperationNode(
                span=SourceSpan(operator.span.start, operand.span.end),
                operator=(
                    UnaryOperator.PLUS
                    if operator.type is TokenType.PLUS
                    else UnaryOperator.MINUS
                ),
                operand=operand,
            )
        return self._parse_power()

    def _parse_power(self) -> ExpressionNode:
        base = self._parse_atom()
        if self._match(TokenType.POWER):
            exponent = self._parse_signed_number_node()
            return PowerNode(
                span=SourceSpan(base.span.start, exponent.span.end),
                base=base,
                exponent=exponent,
            )
        return base

    def _parse_implicit_factor(self) -> ExpressionNode:
        if self._current().type not in _IMPLICIT_FACTOR_STARTERS:
            self._error(
                code="SYN040",
                message="Se esperaba un factor valido para multiplicacion implicita.",
                expected=("VARIABLE", "FRAC", "SQRT", "LBRACE"),
            )

        base = self._parse_atom_non_numeric()
        if self._match(TokenType.POWER):
            exponent = self._parse_signed_number_node()
            return PowerNode(
                span=SourceSpan(base.span.start, exponent.span.end),
                base=base,
                exponent=exponent,
            )
        return base

    def _parse_atom(self) -> ExpressionNode:
        current = self._current()

        if current.type is TokenType.VARIABLE:
            token = self._advance()
            return VariableReferenceNode(span=token.span, name=str(token.value))

        if current.type is TokenType.NUMBER:
            token = self._advance()
            return self._number_node(token)

        if current.type is TokenType.FRAC:
            return self._parse_fraction()

        if current.type is TokenType.SQRT:
            return self._parse_sqrt()

        if current.type is TokenType.LBRACE:
            return self._parse_group()

        self._error(
            code="SYN041",
            message="Se esperaba una expresion matematica.",
            expected=("VARIABLE", "NUMBER", "FRAC", "SQRT", "LBRACE"),
        )

    def _parse_atom_non_numeric(self) -> ExpressionNode:
        current = self._current()

        if current.type is TokenType.VARIABLE:
            token = self._advance()
            return VariableReferenceNode(span=token.span, name=str(token.value))
        if current.type is TokenType.FRAC:
            return self._parse_fraction()
        if current.type is TokenType.SQRT:
            return self._parse_sqrt()
        if current.type is TokenType.LBRACE:
            return self._parse_group()

        self._error(
            code="SYN042",
            message="Un factor implicito no puede comenzar con NUMBER.",
            expected=("VARIABLE", "FRAC", "SQRT", "LBRACE"),
        )

    def _parse_fraction(self) -> FractionNode:
        start = self._consume(TokenType.FRAC, "Se esperaba \\frac.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' para iniciar el numerador de \\frac.")
        numerator = self._parse_expression()
        self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar el numerador de \\frac.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' para iniciar el denominador de \\frac.")
        denominator = self._parse_expression()
        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar el denominador de \\frac.")

        return FractionNode(
            span=SourceSpan(start.span.start, closing.span.end),
            numerator=numerator,
            denominator=denominator,
        )

    def _parse_sqrt(self) -> SqrtNode:
        start = self._consume(TokenType.SQRT, "Se esperaba \\sqrt.")
        self._consume(TokenType.LBRACE, "Se esperaba '{' despues de \\sqrt.")
        expression = self._parse_expression()
        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar \\sqrt.")
        return SqrtNode(
            span=SourceSpan(start.span.start, closing.span.end),
            expression=expression,
        )

    def _parse_group(self) -> ExpressionNode:
        opening = self._consume(TokenType.LBRACE, "Se esperaba '{'.")
        expression = self._parse_expression()
        closing = self._consume(TokenType.RBRACE, "Se esperaba '}' para cerrar la expresion agrupada.")
        return replace(expression, span=SourceSpan(opening.span.start, closing.span.end))

    def _parse_signed_decimal(self) -> tuple[Decimal, SourceSpan]:
        sign = None
        if self._current().type in {TokenType.PLUS, TokenType.MINUS}:
            sign = self._advance()

        number = self._consume(TokenType.NUMBER, "Se esperaba un NUMBER.")
        value = self._number_value(number)
        if sign is not None and sign.type is TokenType.MINUS:
            value = -value

        start = sign.span.start if sign is not None else number.span.start
        return value, SourceSpan(start, number.span.end)

    def _parse_signed_number_node(self) -> NumberLiteralNode | UnaryOperationNode:
        sign = None
        if self._current().type in {TokenType.PLUS, TokenType.MINUS}:
            sign = self._advance()

        number_token = self._consume(TokenType.NUMBER, "El exponente debe ser un NUMBER con signo opcional.")
        number = self._number_node(number_token)
        if sign is None:
            return number

        return UnaryOperationNode(
            span=SourceSpan(sign.span.start, number.span.end),
            operator=(UnaryOperator.PLUS if sign.type is TokenType.PLUS else UnaryOperator.MINUS),
            operand=number,
        )

    @staticmethod
    def _number_value(token: Token) -> Decimal:
        if not isinstance(token.value, Decimal):
            raise TypeError(f"NUMBER token without Decimal value: {token!r}")
        return token.value

    def _number_node(self, token: Token) -> NumberLiteralNode:
        return NumberLiteralNode(
            span=token.span,
            lexeme=token.lexeme,
            value=self._number_value(token),
        )

    def _current(self) -> Token:
        return self.tokens[self._index]

    def _advance(self) -> Token:
        token = self._current()
        if token.type is not TokenType.EOF:
            self._index += 1
        return token

    def _match(self, token_type: TokenType) -> bool:
        if self._current().type is token_type:
            self._advance()
            return True
        return False

    def _consume(self, token_type: TokenType, message: str) -> Token:
        current = self._current()
        if current.type is token_type:
            return self._advance()

        self._error(
            code="SYN001",
            message=message,
            expected=(token_type.name,),
        )

    def _error(
        self,
        *,
        code: str,
        message: str,
        expected: tuple[str, ...] = (),
    ) -> None:
        current = self._current()
        found = current.type.name
        lexeme = current.lexeme or None
        raise CompilationError(
            Diagnostic(
                code=code,
                phase=DiagnosticPhase.SYNTACTIC,
                message=message,
                span=current.span,
                lexeme=lexeme,
                expected=expected,
                found=found,
            )
        )


def parse_tokens(tokens: Iterable[Token]) -> ProgramNode:
    return Parser(tokens).parse()


def parse(source: str) -> ProgramNode:
    return parse_tokens(tokenize(source))
