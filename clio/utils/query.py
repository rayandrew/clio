import sys
from pathlib import Path

from evalidate import CompilationException, ExecutionException, Expr, ValidationException, base_eval_model

from clio.utils.logging import log_create

log = log_create(name=__name__)


def setup_query_model():
    model = base_eval_model.clone()
    model.nodes += [
        "Call",
        "Attribute",
    ]
    model.allowed_functions += [
        "len",
        "sorted",
    ]
    model.imported_functions["minutes"] = minutes
    return model


class Query(Expr):
    def __init__(self, expr: str, filename: str | Path | None = None):
        model = setup_query_model()
        super().__init__(expr, model=model, filename=filename)

    def __call__(self, ctx=None):
        return self.eval(ctx=ctx)


QueryValidationException = ValidationException
QueryCompilationException = CompilationException
QueryExecutionException = ExecutionException


def get_query(expr: str | None = None, filename: str | Path | None = None):
    try:
        return Query(expr, filename=filename) if expr else None
    except (QueryValidationException, QueryCompilationException) as e:
        log.exception(e)
        sys.exit(1)


#### PREDEFINED FUNCTIONS ####


def minutes(x: int | float) -> float:
    """
    Convert ts_record (ms) to minutes
    """
    return float(x / (1000 * 60))


__all__ = ["setup_query_model", "Query", "QueryValidationException", "QueryCompilationException", "QueryExecutionException", "get_query"]
