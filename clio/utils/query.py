import sys
from pathlib import Path

from evalidate import CompilationException, ExecutionException, Expr, ValidationException, base_eval_model

from clio.utils.logging import log_create

log = log_create(name=__name__)


def setup_query_model():
    model = base_eval_model.clone()
    model.nodes += ["Call", "Attribute", "List"]
    model.allowed_functions += [
        "len",
        "sorted",
    ]
    model.imported_functions["minutes"] = minutes
    model.imported_functions["is_substr_in_list"] = is_substr_in_list
    model.imported_functions["is_substr_list_in_str"] = is_substr_list_in_str
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


def is_substr_in_list(substring: str, list_: list[str]) -> bool:
    """
    Check if substring is in list
    """
    return any(substring in s for s in list_)


def is_substr_list_in_str(substring_list: list[str], string: str) -> bool:
    """
    Check if any substring in substring_list is in string
    """
    return any(substring in string for substring in substring_list)


__all__ = ["setup_query_model", "Query", "QueryValidationException", "QueryCompilationException", "QueryExecutionException", "get_query"]
