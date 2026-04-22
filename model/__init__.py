from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "Model",
    "PointAgentsModel",
    "SimpleQueryModel",
    "PROJECT_ROOT",
    "LOGS_DIR",
    "configure_file_logger",
    "read_optional_float",
    "read_optional_int",
    "read_thinking_mode",
    "read_optional_timeout",
]

if TYPE_CHECKING:
    from model.model import Model
    from model.model import (
        LOGS_DIR,
        PROJECT_ROOT,
        configure_file_logger,
        read_optional_float,
        read_optional_int,
        read_optional_timeout,
        read_thinking_mode,
    )
    from model.point_agents_model import PointAgentsModel
    from model.simple_query_model import SimpleQueryModel


def __getattr__(name: str):
    if name == "Model":
        return import_module("model.model").Model
    if name == "PROJECT_ROOT":
        return import_module("model.model").PROJECT_ROOT
    if name == "LOGS_DIR":
        return import_module("model.model").LOGS_DIR
    if name == "configure_file_logger":
        return import_module("model.model").configure_file_logger
    if name == "read_optional_float":
        return import_module("model.model").read_optional_float
    if name == "read_optional_int":
        return import_module("model.model").read_optional_int
    if name == "read_thinking_mode":
        return import_module("model.model").read_thinking_mode
    if name == "read_optional_timeout":
        return import_module("model.model").read_optional_timeout
    if name == "PointAgentsModel":
        return import_module("model.point_agents_model").PointAgentsModel
    if name == "SimpleQueryModel":
        return import_module("model.simple_query_model").SimpleQueryModel
    raise AttributeError(f"module 'model' has no attribute {name!r}")
