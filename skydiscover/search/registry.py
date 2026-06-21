"""
Registries and factory functions for search components.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

from skydiscover.config import Config, DatabaseConfig, build_output_dir, load_config
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.utils.discovery_utils import load_database_from_file
from skydiscover.utils.code_utils import extract_solution_language

logger = logging.getLogger(__name__)


######################### REGISTRY #########################

_PROGRAM_REGISTRY: Dict[str, Type[Program]] = {}
_DATABASE_REGISTRY: Dict[str, Type[ProgramDatabase]] = {}
_CONTROLLER_REGISTRY: Dict[str, Type[DiscoveryController]] = {}


def register_program(search_type: str, program_class: Type[Program]) -> None:
    """Register a program class for a search type."""
    _PROGRAM_REGISTRY[search_type] = program_class
    logger.debug(
        f"Registered program class '{program_class.__name__}' for search type '{search_type}'"
    )


def register_database(search_type: str, database_class: Type[ProgramDatabase]) -> None:
    """Register a database class for a search type."""
    _DATABASE_REGISTRY[search_type] = database_class
    logger.debug(
        f"Registered database class '{database_class.__name__}' for search type '{search_type}'"
    )


def register_controller(search_type: str, controller_class: Type[DiscoveryController]) -> None:
    """Register a discovery controller class for a search type."""
    _CONTROLLER_REGISTRY[search_type] = controller_class
    logger.debug(
        f"Registered controller class '{controller_class.__name__}' for search type '{search_type}'"
    )


######################### FACTORY FUNCTIONS #########################


def create_database(search_type: str, config: DatabaseConfig) -> ProgramDatabase:
    """
    Create a database instance for a given search type.

    Supports both registered search types and dynamic loading for "evox"/"evolve" types
    when a custom database_file_path is specified.
    """
    if search_type in ("evox", "evolve") and getattr(config, "database_file_path", None):
        database_class, program_class = load_database_from_file(config.database_file_path)
        db = database_class(search_type, config)
        db._program_class = program_class
        return db

    if search_type not in _DATABASE_REGISTRY:
        available_types = ", ".join(sorted(_DATABASE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown search type: '{search_type}'. "
            f"Available types: {available_types}. "
            f"For 'evox'/'evolve' type with custom database, set config.search.database.database_file_path"
        )

    database_class = _DATABASE_REGISTRY[search_type]
    return database_class(search_type, config)


def get_program(
    config: Config,
    initial_program_solution: str,
    initial_program_id: str,
    initial_metrics: Dict[str, Any],
    start_iteration: int,
) -> Program:
    """
    Create an initial Program instance appropriate to the search type.

    Supports both registered program classes and dynamic loading for "evox" type
    when a custom database_file_path is specified.
    """
    search_type = config.search.type

    if search_type == "evox" and getattr(config.search.database, "database_file_path", None):
        logger.info(f"Using search strategy from: {config.search.database.database_file_path}")
        _, program_class = load_database_from_file(config.search.database.database_file_path)
        return program_class(
            id=initial_program_id,
            solution=initial_program_solution,
            language=config.language,
            metrics=initial_metrics,
            iteration_found=start_iteration,
        )

    program_class = _PROGRAM_REGISTRY.get(search_type, Program)
    return program_class(
        id=initial_program_id,
        solution=initial_program_solution,
        language=config.language,
        metrics=initial_metrics,
        iteration_found=start_iteration,
    )


def setup_search(
    initial_program_path: str,
    evaluation_file: str,
    config_path: str,
    output_dir: Optional[str] = None,
    evaluator_env_vars: Optional[Dict[str, str]] = None,
    parent_llm_config: Optional["LLMConfig"] = None,
    share_llm: bool = False,
) -> Tuple[DiscoveryControllerInput, str]:
    """
    Load config, create database, and build a DiscoveryControllerInput from a config path.

    This is the lightweight alternative to creating a full Runner when
    you only need the config/database/controller (e.g. for the search side of
    co-evolution).

    Args:
        parent_llm_config: The parent (main discovery) LLM config. Used to
            inherit the parent endpoint/models when share_llm is True, and to
            detect a parent/meta provider mismatch (logged as a warning) when it
            is False.
        share_llm: If True, the meta-search inherits the parent's LLM endpoint,
            models, and credentials (from search.share_llm). If False (default),
            the meta-search keeps its own configured provider (the bundled
            search.yaml ships an OpenAI default); when that provider differs from
            the parent's, a warning is logged so the mismatch is not silent.

    Returns:
        Tuple of (controller_input, initial_program_solution)
    """
    config = load_config(config_path)

    # The meta-search keeps its own configured provider by default (the bundled
    # search.yaml ships an OpenAI default). Only inherit the parent endpoint when
    # share_llm is explicitly set. Use the parent's actual model configs (which
    # carry the correct per-model api_base/api_key, e.g. Azure endpoints) rather
    # than the top-level LLMConfig defaults which may still point to openai.com.
    if share_llm and parent_llm_config is not None and parent_llm_config.models:
        import copy

        parent_models = [copy.deepcopy(m) for m in parent_llm_config.models]
        config.llm.models = parent_models
        config.llm.evaluator_models = [copy.deepcopy(m) for m in parent_llm_config.models]
        config.llm.guide_models = [copy.deepcopy(m) for m in parent_llm_config.models]
        # Sync top-level api_base/api_key from the first parent model
        config.llm.api_base = parent_models[0].api_base or config.llm.api_base
        config.llm.api_key = parent_models[0].api_key or config.llm.api_key
    elif parent_llm_config is not None:
        # Not sharing: warn loudly if the meta-search will run against a
        # different provider than the main discovery process, instead of letting
        # it silently fall back to its configured default (openai.com) and fail
        # only later, mid-run, on the first call.
        parent_base = (
            parent_llm_config.models[0].api_base
            if parent_llm_config.models and parent_llm_config.models[0].api_base
            else parent_llm_config.api_base
        )
        meta_base = (
            config.llm.models[0].api_base
            if config.llm.models and config.llm.models[0].api_base
            else config.llm.api_base
        )
        if parent_base and meta_base and parent_base != meta_base:
            logger.warning(
                "EvoX meta-search is configured to use LLM endpoint %s (from %s), "
                "which differs from the main discovery process endpoint %s. The "
                "meta-search may fail at runtime if that endpoint/model is not "
                "reachable. To run it on the same endpoint set search.share_llm: "
                "true (optionally add llm.guide_models in %s for a cheaper guide); "
                "to use a different provider on purpose, declare llm.models in %s "
                "to silence this warning.",
                meta_base,
                config_path,
                parent_base,
                config_path,
                config_path,
            )

    with open(initial_program_path, "r") as f:
        initial_program_solution = f.read()

    if not config.language:
        config.language = extract_solution_language(initial_program_solution)

    file_extension = os.path.splitext(initial_program_path)[1] or ".py"
    if not file_extension.startswith("."):
        file_extension = f".{file_extension}"
    if config.file_suffix == ".py":
        config.file_suffix = file_extension

    database = create_database(config.search.type, config.search.database)

    if not output_dir:
        output_dir = build_output_dir(config.search.type, initial_program_path)

    controller_input = DiscoveryControllerInput(
        config=config,
        evaluation_file=evaluation_file,
        database=database,
        file_suffix=config.file_suffix,
        output_dir=output_dir,
        evaluator_env_vars=evaluator_env_vars,
    )
    return controller_input, initial_program_solution
