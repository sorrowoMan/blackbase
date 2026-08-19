"""
Canonical context keys used across adapters/plugins/biases in both NSGABlack and MLBlack.

The goal is consistency: different modules can interoperate by reading/writing
the same keys. This file is intentionally small and stable.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Generic keys (shared between NSGABlack and MLBlack)
# ---------------------------------------------------------------------------
KEY_TEMPERATURE = "temperature"
KEY_GENERATION = "generation"
KEY_STEP = "step"
KEY_PROBLEM = "problem"
KEY_POPULATION = "population"
KEY_OBJECTIVES = "objectives"
KEY_INDIVIDUAL = "individual"
KEY_BEST_X = "best_x"
KEY_BEST_OBJECTIVE = "best_objective"
KEY_HISTORY = "history"
KEY_METADATA = "metadata"
KEY_METADATA_LAYERS = "metadata.layers"
KEY_PROBLEM_DATA = "problem_data"
KEY_CONSTRAINT_VIOLATION = "constraint_violation"
KEY_CONSTRAINT_VIOLATIONS = "constraint_violations"
KEY_CONSTRAINTS = "constraints"
KEY_INDIVIDUAL_ID = "individual_id"
KEY_BOUNDS = "bounds"
KEY_CAPACITY = "capacity"
KEY_SHAPE = "shape"

# ---------------------------------------------------------------------------
# NSGABlack-specific keys (optimization-focused)
# ---------------------------------------------------------------------------
KEY_NUM_NODES = "num_nodes"
KEY_DISTANCE_MATRIX = "distance_matrix"
KEY_ROW_SUMS = "row_sums"
KEY_COL_SUMS = "col_sums"
KEY_K_NONZERO = "k_nonzero"
KEY_DENSITY = "density"
KEY_BLOCK_MIN = "block_min"
KEY_BLOCK_MAX = "block_max"
KEY_MUTATION_SIGMA = "mutation_sigma"
KEY_VNS_K = "vns_k"
KEY_STRATEGY = "strategy"
KEY_STRATEGY_ID = "strategy_id"
KEY_SHARED = "shared"
KEY_ROLE = "role"
KEY_ROLE_INDEX = "role_index"
KEY_ROLE_ADAPTER = "role_adapter"
KEY_TASK = "task"
KEY_REPORT = "report"
KEY_ROLE_REPORTS = "role_reports"
KEY_CANDIDATE_ROLES = "candidate_roles"
KEY_CANDIDATE_UNITS = "candidate_units"
KEY_UNIT_TASKS = "unit_tasks"
KEY_ADAPTER_NAME = "adapter_name"
KEY_ADAPTER_CURRENT_SCORE = "adapter_current_score"
KEY_ADAPTER_BEST_SCORE = "adapter_best_score"
KEY_ADAPTER_BEST_X = "adapter_best_x"
KEY_ADAPTER_BEST_OBJECTIVES = "adapter_best_objectives"
KEY_RUNTIME_PROJECTION_AUDIT = "runtime_projection_audit"
KEY_DYNAMIC = "dynamic"
KEY_PHASE_ID = "phase_id"
KEY_COMPANION_PHASE_INDEX = "companion_phase_index"
KEY_COMPANION_TRIGGER_REASON = "companion_trigger_reason"
KEY_COMPANION_NEXT_ELIGIBLE_GENERATION = "companion_next_eligible_generation"
KEY_COMPANION_PHASE_COUNT_USED = "companion_phase_count_used"
KEY_STAGE_INDEX = "stage_index"
KEY_STAGE_NAME = "stage_name"
KEY_STAGE_TOTAL = "stage_total"
KEY_STAGE_ARTIFACTS = "stage_artifacts"
KEY_STAGE_ARTIFACT_PREFIX = "stage_artifact."
KEY_STAGE_STATUS = "stage_status"
KEY_STAGE_INPUT_ARTIFACTS = "stage_input_artifacts"
KEY_STAGE_OUTPUT_ARTIFACTS = "stage_output_artifacts"
KEY_PHASE = "phase"
KEY_REGION_ID = "region_id"
KEY_REGION_BOUNDS = "region_bounds"
KEY_SEEDS = "seeds"
KEY_RUNNING = "running"
KEY_EVENT_QUEUE = "event_queue"
KEY_EVENT_INFLIGHT = "event_inflight"
KEY_EVENT_ARCHIVE = "event_archive"
KEY_EVENT_HISTORY = "event_history"
KEY_EVENT_SHARED = "event_shared"
KEY_SINGLE_TRAJ_STATE = "single_traj_state"
KEY_SINGLE_TRAJ_SIGMA = "single_traj_sigma"
KEY_MOEAD_SUBPROBLEM = "moead_subproblem"
KEY_MOEAD_WEIGHT = "moead_weight"
KEY_MOEAD_NEIGHBOR_MODE = "moead_neighbor_mode"
KEY_MO_WEIGHTS = "mo_weights"
KEY_MAS_MODEL = "mas_model"
KEY_SUBSPACE_BASIS = "subspace_basis"

# ---------------------------------------------------------------------------
# MLBlack-specific keys (machine learning-focused)
# ---------------------------------------------------------------------------
KEY_ADAPTER_BEST_STATE = "adapter.best_state"
KEY_ADAPTER_CURRENT_STATE = "adapter.current_state"
KEY_ADAPTER_SEARCH_STATE = "adapter.search_state"
KEY_AUTOGRAD_OPTIM_CONFIG = "autograd.optim.config"
KEY_ADAPTER_STATE = "adapter.state"
KEY_ARTIFACT_MODEL = "artifact.model"
KEY_ARTIFACT_REPORT = "artifact.report"
KEY_ARTIFACT_STATE = "artifact.state"
KEY_ARTIFACT_SYMBOLIC_BASIS_REF = "artifact.symbolic_basis_ref"
KEY_ARTIFACT_SYMBOLIC_TASK_REF = "artifact.symbolic_task_ref"
KEY_ARTIFACT_VIEWER = "artifact.viewer"
KEY_BASE_DECODER = "base_decoder"
KEY_BACKEND_CAPABILITY = "backend.capability"
KEY_BACKEND_CONTRACT = "backend.contract"
KEY_BACKEND_DEVICE = "backend.device"
KEY_BACKEND_DEVICE_POLICY = "backend.device_policy"
KEY_BACKEND_NAME = "backend.name"
KEY_BACKEND_REQUESTED_NAME = "backend.requested_name"
KEY_BACKEND_SESSION = "backend.session"
KEY_BASIS_ARTIFACT_REF = "basis.artifact_ref"
KEY_BASIS_CANDIDATE_REF = "basis.candidate_ref"
KEY_BASIS_CONSENSUS = "basis.consensus"
KEY_BASIS_FITTED_REF = "basis.fitted_ref"
KEY_BASIS_METRICS = "basis.metrics"
KEY_BASIS_OVERLAP_REPORT = "basis.overlap_report"
KEY_BRANCH_SPEC = "branch.spec"
KEY_BIAS_BRANCH = "bias.branch"
KEY_BIAS_DYNAMIC_POOL = "bias.dynamic_pool"
KEY_BIAS_L2_SCALE = "bias.l2_scale"
KEY_BIAS_NOOP = "bias.noop"
KEY_BIAS_OBJECTIVE_POLICY = "bias.objective_policy"
KEY_BIAS_OBJECTIVE_WEIGHTS = "bias.objective_weights"
KEY_BIAS_SOFT_PREFERENCE = "bias.soft_preference"
KEY_BIAS_STATE_L2_PENALTY = "bias.state_l2_penalty"
KEY_BRANCH_REPRESENTATIONS = "branch_representations"
KEY_CAPABILITY_SIDE_EFFECT = "capability.side_effect"
KEY_CANDIDATE_BRANCH = "candidate.branch"
KEY_CANDIDATE_DISTRIBUTION_MODEL = "candidate.distribution_model"
KEY_CANDIDATE_INTERVAL_MODEL = "candidate.interval_model"
KEY_CANDIDATE_MODEL = "candidate.model"
KEY_CANDIDATE_MODEL_SPEC = "candidate.model_spec"
KEY_CANDIDATE_OUTPUT = "candidate.output"
KEY_CANDIDATE_PROBABILITY_MODEL = "candidate.probability_model"
KEY_CANDIDATE_REPAIRED_STATE = "candidate.repaired_state"
KEY_CANDIDATE_SYMBOLIC_BASIS_MODEL = "candidate.symbolic_basis_model"
KEY_CANDIDATE_UNKNOWN_STATE = "candidate.unknown_state"
KEY_CANDIDATE_FORECAST_MODEL = "candidate.forecast_model"
KEY_CHECKPOINT_REF = "checkpoint.ref"
KEY_DATA = "data"
KEY_DATA_FEATURE_NAMES = "data.feature_names"
KEY_DATA_GRAPHS = "data.graphs"
KEY_DATA_IMAGE_PAIRS = "data.image_pairs"
KEY_DATA_IMAGES = "data.images"
KEY_DATA_NUMERIC_VIEW = "data.numeric_view"
KEY_DATA_PREFERENCE_PAIRS = "data.preference_pairs"
KEY_DATA_RAW_ROWS = "data.raw_rows"
KEY_DATA_SCHEMA = "data.schema"
KEY_DATA_TARGET = "data.target"
KEY_DATA_TIME_SERIES_VIEW = "data.time_series_view"
KEY_DATA_X_TRAIN = "data.X_train"
KEY_DATA_X_VALID = "data.X_valid"
KEY_DATA_Y_TRAIN = "data.y_train"
KEY_DATA_Y_VALID = "data.y_valid"
KEY_PRETRAINED_MODEL = "pretrained.model"
KEY_PRETRAINED_CHECKPOINT_MAP = "pretrained.checkpoint_map"
KEY_PRETRAINED_CHECKPOINT_REPORT = "pretrained.checkpoint_report"
KEY_PRETRAINED_TOKENIZER = "pretrained.tokenizer"
KEY_ESTIMATOR_FACTORY = "estimator.factory"
KEY_EVENT_DECISION = "event.decision"
KEY_EXPERIMENT_RECORDS = "experiment.records"
KEY_FEEDBACK_CONSTRAINTS = "feedback.constraints"
KEY_FEEDBACK_GRADIENTS = "feedback.gradients"
KEY_FEEDBACK_LOSS = "feedback.loss"
KEY_FEEDBACK_METRICS = "feedback.metrics"
KEY_FEEDBACK_OBJECTIVES = "feedback.objectives"
KEY_FEEDBACK_RESIDUALS = "feedback.residuals"
KEY_FEEDBACK_SIGNALS = "feedback.signals"
KEY_FITTED_ESTIMATOR = "fitted_estimator"
KEY_HEAD_OUTPUT = "head.output"
KEY_MODEL_PARAMETER_GRADIENT = "model.parameter_gradient"
KEY_MODEL_PREDICT = "model.predict"
KEY_MODEL_PREDICT_PARAMS = "model.predict_params"
KEY_MODEL_PREDICT_INTERVAL = "model.predict_interval"
KEY_MODEL_PREDICT_PROBA = "model.predict_proba"
KEY_MODEL_ROUTE = "model.route"
KEY_MODEL_TRANSFORM = "model.transform"
KEY_MODEL_LOGITS = "model.logits"
KEY_MODEL_HIDDEN_STATES = "model.hidden_states"
KEY_MODEL_EMBEDDINGS = "model.embeddings"
KEY_MODEL_RANKING_SCORES = "model.ranking_scores"
KEY_MODEL_PREFERENCE_SCORES = "model.preference_scores"
KEY_NEURAL_GRAPH_SPEC = "neural.graph_spec"
KEY_NEURAL_PARAMETER_LAYOUT = "neural.parameter_layout"
KEY_NEURAL_OPTIMIZER_STATE = "neural.optimizer_state"
KEY_NEURAL_TRANSFORMER_SPEC = "neural.transformer_spec"
KEY_NEURAL_HIDDEN_STATES = "neural.hidden_states"
KEY_NEURAL_ATTENTION_MAPS = "neural.attention_maps"
KEY_NEURAL_FFN_ACTIVATIONS = "neural.ffn_activations"
KEY_NEURAL_AUDIT = "neural.audit"
KEY_NEURAL_AUDIT_ATTENTION_SUMMARY = "neural.audit.attention_summary"
KEY_NEURAL_AUDIT_ATTENTION_HEAD_CORR = "neural.audit.attention_head_corr"
KEY_NEURAL_AUDIT_FFN_SUMMARY = "neural.audit.ffn_summary"
KEY_NEURAL_AUDIT_FFN_ACTIVATION_SPARSITY = "neural.audit.ffn_activation_sparsity"
KEY_ORTHOGONAL_FEATURE_MAP = "orthogonal_feature_map"
KEY_PIPELINE_FEATURE_SPACE = "pipeline.feature_space"
KEY_PIPELINE_COMPONENT_STATE = "pipeline.component_state"
KEY_PIPELINE_CONDITIONAL_FEATURES = "pipeline.conditional_features"
KEY_PIPELINE_FIT_STATE = "pipeline.fit_state"
KEY_PIPELINE_SLOT_CONTEXT = "pipeline.slot_context"
KEY_POPULATION_CANDIDATES = "population.candidates"
KEY_POPULATION_FEEDBACK = "population.feedback"
KEY_POPULATION_SNAPSHOT_REF = "population.snapshot_ref"
KEY_PREFERENCE_REFERENCE_MODEL = "preference.reference_model"
KEY_PROBLEM_DATA_X_TRAIN = "problem.data.X_train"
KEY_PROBLEM_DATA_Y_TRAIN = "problem.data.y_train"
KEY_REPRESENTATION_NUMPY_MLP_POINT = "representation.numpy_mlp_point"
KEY_RESOURCE_AUDIT = "resource.audit"
KEY_RESOURCE_CONTEXT = "resource.context"
KEY_RESOURCE_DEVICE = "resource.device"
KEY_RESOURCE_LEASE = "resource.lease"
KEY_RESOURCE_CONTEXT_SHORT = "resource_context"
KEY_ROUTER = "router"
KEY_SIGNAL_POOL = "signal.pool"
KEY_SIGNAL_BUDGET_REMAINING_RATIO = "signal.budget.remaining_ratio"
KEY_SIGNAL_GATE_ENABLED = "signal.gate.enabled"
KEY_SNAPSHOT_REF = "snapshot.ref"
KEY_STAGE_AUDIT = "stage.audit"
KEY_STAGE_ID = "stage.id"
KEY_SYMBOLIC_ARTIFACT = "symbolic.artifact"
KEY_SYMBOLIC_ARTIFACT_SCHEMA = "symbolic.artifact_schema"
KEY_SYMBOLIC_BRANCH_REPORT = "symbolic.branch_report"
KEY_SYMBOLIC_BASIS_MODEL = "symbolic.basis_model"
KEY_SYMBOLIC_CANDIDATE_POOL = "symbolic.candidate_pool"
KEY_SYMBOLIC_CANDIDATE_LINEAGE = "symbolic.candidate_lineage"
KEY_SYMBOLIC_CANDIDATE_SCORE = "symbolic.candidate_score"
KEY_SYMBOLIC_DECODER_SPEC = "symbolic.decoder_spec"
KEY_SYMBOLIC_EQUIVALENCE_REPORT = "symbolic.equivalence_report"
KEY_SYMBOLIC_EVALUATION_EVENTS = "symbolic.evaluation_events"
KEY_SYMBOLIC_EXPRESSION_SPEC = "symbolic.expression_spec"
KEY_SYMBOLIC_FUNCTION_POOL = "symbolic.function_pool"
KEY_SYMBOLIC_FUNCTION_SPACE = "symbolic.function_space"
KEY_SYMBOLIC_FOLD_REPORT = "symbolic.fold_report"
KEY_SYMBOLIC_GENOME = "symbolic.genome"
KEY_SYMBOLIC_GRAPH_CACHE = "symbolic.graph_cache"
KEY_SYMBOLIC_GRADIENT_SIGNAL = "symbolic.gradient_signal"
KEY_SYMBOLIC_NATIVE_STRUCTURE_SCORE = "symbolic.native_structure_score"
KEY_SYMBOLIC_OVERFIT_GUARD = "symbolic.overfit_guard"
KEY_SYMBOLIC_PARAMETER_SPECS = "symbolic.parameter_specs"
KEY_SYMBOLIC_PARAMETER_VALUES = "symbolic.parameter_values"
KEY_SYMBOLIC_PATH_MEMORY = "symbolic.path_memory"
KEY_SYMBOLIC_POOL_DELTA = "symbolic.pool_delta"
KEY_SYMBOLIC_PRIMITIVE_REGISTRY = "symbolic.primitive_registry"
KEY_SYMBOLIC_REPLAY_RECORD = "symbolic.replay_record"
KEY_SYMBOLIC_SEARCH_POLICY = "symbolic.search_policy"
KEY_SYMBOLIC_SEARCH_SPACE = "symbolic.search_space"
KEY_SYMBOLIC_SIMPLIFICATION_TRACE = "symbolic.simplification_trace"
KEY_SYMBOLIC_STRUCTURE_GUARD = "symbolic.structure_guard"
KEY_SYMBOLIC_TRUTH_CONTRACT_RECOVERY = "symbolic.truth_contract_recovery"
KEY_TASK_FITTED_MODEL_REF = "task.fitted_model_ref"
KEY_TASK_METRICS = "task.metrics"
KEY_TEXT_TOKEN_IDS = "text.token_ids"
KEY_TIME_SERIES_DECOMPOSITION = "time_series.decomposition"
KEY_TIME_SERIES_HORIZON = "time_series.horizon"
KEY_TIME_SERIES_MIN_TRAIN_SIZE = "time_series.min_train_size"
KEY_TIME_SERIES_OBJECTIVE_METRICS = "time_series.objective_metrics"
KEY_TIME_SERIES_SEARCH_SPACE = "time_series.search_space"
KEY_TIME_SERIES_VALIDATION_SIZE = "time_series.validation_size"
KEY_TIME_SERIES_WINDOW_CONFIG = "time_series.window_config"
KEY_TOKENIZER_VOCAB = "tokenizer.vocab"
KEY_TRAINER_CONTEXT = "trainer.context"
KEY_TRAINER_GET_STATE = "trainer.get_state"
KEY_TRAINER_REPORT = "trainer.report"
KEY_TRAINER_SNAPSHOT_STORE = "trainer.snapshot_store"
KEY_TRAINER_STEP = "trainer.step"
KEY_TRAINING_RESULT = "training.result"
KEY_TRAINING_TASK = "training.task"

# ---------------------------------------------------------------------------
# Metrics namespace (shared)
# ---------------------------------------------------------------------------
KEY_METRICS = "metrics"
KEY_METRICS_MC_SAMPLES = "metrics.mc_samples"
KEY_METRICS_MC_MEAN = "metrics.mc_mean"
KEY_METRICS_MC_STD = "metrics.mc_std"
KEY_METRICS_MC_MIN = "metrics.mc_min"
KEY_METRICS_MC_MAX = "metrics.mc_max"
KEY_METRICS_SURROGATE_STD = "metrics.surrogate_std"
KEY_METRICS_IMPLICIT_RESIDUAL = "metrics.implicit_residual"
KEY_METRICS_IMPLICIT_ITERS = "metrics.implicit_iters"
KEY_METRICS_IMPLICIT_SUCCESS = "metrics.implicit_success"
KEY_METRICS_INNER_ELAPSED_MS = "metrics.inner_elapsed_ms"
KEY_METRICS_INNER_STATUS = "metrics.inner_status"
KEY_METRICS_INNER_CALLS = "metrics.inner_calls"
KEY_METRICS_SOFT_ERROR_COUNT = "metrics.soft_error_count"
KEY_METRICS_SOFT_ERROR_LAST = "metrics.soft_error_last"

# ---------------------------------------------------------------------------
# Context meta (shared)
# ---------------------------------------------------------------------------
KEY_EVALUATION_COUNT = "evaluation_count"
KEY_PARETO_SOLUTIONS = "pareto_solutions"
KEY_PARETO_OBJECTIVES = "pareto_objectives"
KEY_MUTATION_RATE = "mutation_rate"
KEY_CROSSOVER_RATE = "crossover_rate"
KEY_CONTEXT_SCHEMA = "context_schema"
KEY_CONTEXT_EVENTS = "context_events"
KEY_CONTEXT_CACHE = "context_cache"
KEY_DECISION_TRACE = "decision_trace"
KEY_CHECKPOINT_LATEST_PATH = "checkpoint.latest_path"
KEY_CHECKPOINT_LAST_LOADED_PATH = "checkpoint.last_loaded_path"
KEY_BIAS_CACHE_FINGERPRINT = "bias_cache_fingerprint"
KEY_SNAPSHOT_KEY = "snapshot_key"
KEY_SNAPSHOT_BACKEND = "snapshot_backend"
KEY_SNAPSHOT_SCHEMA = "snapshot_schema"
KEY_SNAPSHOT_META = "snapshot_meta"
KEY_BEST_CANDIDATE_REF = "best_candidate_ref"
KEY_POPULATION_REF = "population_ref"
KEY_OBJECTIVES_REF = "objectives_ref"
KEY_CONSTRAINT_VIOLATIONS_REF = "constraint_violations_ref"
KEY_PARETO_SOLUTIONS_REF = "pareto_solutions_ref"
KEY_PARETO_OBJECTIVES_REF = "pareto_objectives_ref"
KEY_HISTORY_REF = "history_ref"
KEY_DECISION_TRACE_REF = "decision_trace_ref"
KEY_SEQUENCE_GRAPH_REF = "sequence_graph_ref"
KEY_BACKEND_WARM_START_REF = "backend_warm_start_ref"
KEY_BACKEND_SOLUTION_POOL_REF = "backend_solution_pool_ref"
KEY_BACKEND_DIAGNOSTIC_REF = "backend_diagnostic_ref"
KEY_BACKEND_CALLBACK_REF = "backend_callback_ref"

# ---------------------------------------------------------------------------
# Unified registry
# ---------------------------------------------------------------------------
CANONICAL_CONTEXT_KEYS = {
    # Generic
    KEY_TEMPERATURE, KEY_GENERATION, KEY_STEP, KEY_PROBLEM, KEY_POPULATION,
    KEY_OBJECTIVES, KEY_INDIVIDUAL, KEY_BEST_X, KEY_BEST_OBJECTIVE, KEY_HISTORY,
    KEY_METADATA, KEY_METADATA_LAYERS, KEY_PROBLEM_DATA, KEY_CONSTRAINT_VIOLATION,
    KEY_CONSTRAINT_VIOLATIONS, KEY_CONSTRAINTS, KEY_INDIVIDUAL_ID, KEY_BOUNDS,
    KEY_CAPACITY, KEY_SHAPE,
    
    # NSGABlack-specific
    KEY_NUM_NODES, KEY_DISTANCE_MATRIX, KEY_ROW_SUMS, KEY_COL_SUMS, KEY_K_NONZERO,
    KEY_DENSITY, KEY_BLOCK_MIN, KEY_BLOCK_MAX, KEY_MUTATION_SIGMA, KEY_VNS_K,
    KEY_STRATEGY, KEY_STRATEGY_ID, KEY_SHARED, KEY_ROLE, KEY_ROLE_INDEX,
    KEY_ROLE_ADAPTER, KEY_TASK, KEY_REPORT, KEY_ROLE_REPORTS, KEY_CANDIDATE_ROLES,
    KEY_CANDIDATE_UNITS, KEY_UNIT_TASKS, KEY_ADAPTER_NAME, KEY_ADAPTER_CURRENT_SCORE,
    KEY_ADAPTER_BEST_SCORE, KEY_ADAPTER_BEST_X, KEY_ADAPTER_BEST_OBJECTIVES,
    KEY_RUNTIME_PROJECTION_AUDIT,
    KEY_DYNAMIC, KEY_PHASE_ID, KEY_COMPANION_PHASE_INDEX,
    KEY_COMPANION_TRIGGER_REASON, KEY_COMPANION_NEXT_ELIGIBLE_GENERATION,
    KEY_COMPANION_PHASE_COUNT_USED, KEY_STAGE_INDEX, KEY_STAGE_NAME, KEY_STAGE_TOTAL,
    KEY_STAGE_ARTIFACTS, KEY_STAGE_ARTIFACT_PREFIX, KEY_STAGE_STATUS,
    KEY_STAGE_INPUT_ARTIFACTS, KEY_STAGE_OUTPUT_ARTIFACTS, KEY_PHASE, KEY_REGION_ID,
    KEY_REGION_BOUNDS, KEY_SEEDS, KEY_RUNNING, KEY_EVENT_QUEUE, KEY_EVENT_INFLIGHT,
    KEY_EVENT_ARCHIVE, KEY_EVENT_HISTORY, KEY_EVENT_SHARED, KEY_SINGLE_TRAJ_STATE,
    KEY_SINGLE_TRAJ_SIGMA, KEY_MOEAD_SUBPROBLEM, KEY_MOEAD_WEIGHT,
    KEY_MOEAD_NEIGHBOR_MODE, KEY_MO_WEIGHTS, KEY_MAS_MODEL, KEY_SUBSPACE_BASIS,
    
    # MLBlack-specific
    KEY_ADAPTER_BEST_STATE, KEY_ADAPTER_CURRENT_STATE, KEY_ADAPTER_SEARCH_STATE,
    KEY_AUTOGRAD_OPTIM_CONFIG, KEY_ADAPTER_STATE, KEY_ARTIFACT_MODEL,
    KEY_ARTIFACT_REPORT, KEY_ARTIFACT_STATE, KEY_ARTIFACT_SYMBOLIC_BASIS_REF,
    KEY_ARTIFACT_SYMBOLIC_TASK_REF, KEY_ARTIFACT_VIEWER, KEY_BASE_DECODER,
    KEY_BACKEND_CAPABILITY, KEY_BACKEND_CONTRACT, KEY_BACKEND_DEVICE,
    KEY_BACKEND_DEVICE_POLICY, KEY_BACKEND_NAME, KEY_BACKEND_REQUESTED_NAME,
    KEY_BACKEND_SESSION, KEY_BASIS_ARTIFACT_REF, KEY_BASIS_CANDIDATE_REF,
    KEY_BASIS_CONSENSUS, KEY_BASIS_FITTED_REF, KEY_BASIS_METRICS,
    KEY_BASIS_OVERLAP_REPORT, KEY_BRANCH_SPEC, KEY_BIAS_BRANCH, KEY_BIAS_DYNAMIC_POOL,
    KEY_BIAS_L2_SCALE, KEY_BIAS_NOOP, KEY_BIAS_OBJECTIVE_POLICY,
    KEY_BIAS_OBJECTIVE_WEIGHTS, KEY_BIAS_SOFT_PREFERENCE, KEY_BIAS_STATE_L2_PENALTY,
    KEY_BRANCH_REPRESENTATIONS, KEY_CAPABILITY_SIDE_EFFECT, KEY_CANDIDATE_BRANCH,
    KEY_CANDIDATE_DISTRIBUTION_MODEL, KEY_CANDIDATE_INTERVAL_MODEL, KEY_CANDIDATE_MODEL,
    KEY_CANDIDATE_MODEL_SPEC, KEY_CANDIDATE_OUTPUT, KEY_CANDIDATE_PROBABILITY_MODEL,
    KEY_CANDIDATE_REPAIRED_STATE, KEY_CANDIDATE_SYMBOLIC_BASIS_MODEL,
    KEY_CANDIDATE_UNKNOWN_STATE, KEY_CANDIDATE_FORECAST_MODEL, KEY_CHECKPOINT_REF,
    KEY_DATA, KEY_DATA_FEATURE_NAMES, KEY_DATA_GRAPHS, KEY_DATA_IMAGE_PAIRS,
    KEY_DATA_IMAGES, KEY_DATA_NUMERIC_VIEW, KEY_DATA_PREFERENCE_PAIRS,
    KEY_DATA_RAW_ROWS, KEY_DATA_SCHEMA, KEY_DATA_TARGET, KEY_DATA_TIME_SERIES_VIEW,
    KEY_DATA_X_TRAIN, KEY_DATA_X_VALID, KEY_DATA_Y_TRAIN, KEY_DATA_Y_VALID,
    KEY_PRETRAINED_MODEL, KEY_PRETRAINED_CHECKPOINT_MAP, KEY_PRETRAINED_CHECKPOINT_REPORT,
    KEY_PRETRAINED_TOKENIZER, KEY_ESTIMATOR_FACTORY, KEY_EVENT_DECISION,
    KEY_EXPERIMENT_RECORDS, KEY_FEEDBACK_CONSTRAINTS, KEY_FEEDBACK_GRADIENTS,
    KEY_FEEDBACK_LOSS, KEY_FEEDBACK_METRICS, KEY_FEEDBACK_OBJECTIVES,
    KEY_FEEDBACK_RESIDUALS, KEY_FEEDBACK_SIGNALS, KEY_FITTED_ESTIMATOR, KEY_HEAD_OUTPUT,
    KEY_MODEL_PARAMETER_GRADIENT, KEY_MODEL_PREDICT, KEY_MODEL_PREDICT_PARAMS,
    KEY_MODEL_PREDICT_INTERVAL, KEY_MODEL_PREDICT_PROBA, KEY_MODEL_ROUTE,
    KEY_MODEL_TRANSFORM, KEY_MODEL_LOGITS, KEY_MODEL_HIDDEN_STATES,
    KEY_MODEL_EMBEDDINGS, KEY_MODEL_RANKING_SCORES, KEY_MODEL_PREFERENCE_SCORES,
    KEY_NEURAL_GRAPH_SPEC, KEY_NEURAL_PARAMETER_LAYOUT, KEY_NEURAL_OPTIMIZER_STATE,
    KEY_NEURAL_TRANSFORMER_SPEC, KEY_NEURAL_HIDDEN_STATES, KEY_NEURAL_ATTENTION_MAPS,
    KEY_NEURAL_FFN_ACTIVATIONS, KEY_NEURAL_AUDIT, KEY_NEURAL_AUDIT_ATTENTION_SUMMARY,
    KEY_NEURAL_AUDIT_ATTENTION_HEAD_CORR, KEY_NEURAL_AUDIT_FFN_SUMMARY,
    KEY_NEURAL_AUDIT_FFN_ACTIVATION_SPARSITY, KEY_ORTHOGONAL_FEATURE_MAP,
    KEY_PIPELINE_FEATURE_SPACE, KEY_PIPELINE_COMPONENT_STATE,
    KEY_PIPELINE_CONDITIONAL_FEATURES, KEY_PIPELINE_FIT_STATE, KEY_PIPELINE_SLOT_CONTEXT,
    KEY_POPULATION_CANDIDATES,
    KEY_POPULATION_FEEDBACK, KEY_POPULATION_SNAPSHOT_REF, KEY_PREFERENCE_REFERENCE_MODEL,
    KEY_PROBLEM_DATA_X_TRAIN, KEY_PROBLEM_DATA_Y_TRAIN, KEY_REPRESENTATION_NUMPY_MLP_POINT,
    KEY_RESOURCE_AUDIT, KEY_RESOURCE_CONTEXT, KEY_RESOURCE_DEVICE, KEY_RESOURCE_LEASE,
    KEY_RESOURCE_CONTEXT_SHORT, KEY_ROUTER, KEY_SIGNAL_POOL,
    KEY_SIGNAL_BUDGET_REMAINING_RATIO, KEY_SIGNAL_GATE_ENABLED, KEY_SNAPSHOT_REF,
    KEY_STAGE_AUDIT, KEY_STAGE_ID, KEY_SYMBOLIC_ARTIFACT, KEY_SYMBOLIC_ARTIFACT_SCHEMA,
    KEY_SYMBOLIC_BRANCH_REPORT, KEY_SYMBOLIC_BASIS_MODEL, KEY_SYMBOLIC_CANDIDATE_POOL,
    KEY_SYMBOLIC_CANDIDATE_LINEAGE, KEY_SYMBOLIC_CANDIDATE_SCORE,
    KEY_SYMBOLIC_DECODER_SPEC, KEY_SYMBOLIC_EQUIVALENCE_REPORT,
    KEY_SYMBOLIC_EVALUATION_EVENTS, KEY_SYMBOLIC_EXPRESSION_SPEC,
    KEY_SYMBOLIC_FUNCTION_POOL, KEY_SYMBOLIC_FUNCTION_SPACE, KEY_SYMBOLIC_FOLD_REPORT,
    KEY_SYMBOLIC_GENOME, KEY_SYMBOLIC_GRAPH_CACHE, KEY_SYMBOLIC_GRADIENT_SIGNAL,
    KEY_SYMBOLIC_NATIVE_STRUCTURE_SCORE, KEY_SYMBOLIC_OVERFIT_GUARD,
    KEY_SYMBOLIC_PARAMETER_SPECS, KEY_SYMBOLIC_PARAMETER_VALUES, KEY_SYMBOLIC_PATH_MEMORY,
    KEY_SYMBOLIC_POOL_DELTA, KEY_SYMBOLIC_PRIMITIVE_REGISTRY, KEY_SYMBOLIC_REPLAY_RECORD,
    KEY_SYMBOLIC_SEARCH_POLICY, KEY_SYMBOLIC_SEARCH_SPACE, KEY_SYMBOLIC_SIMPLIFICATION_TRACE,
    KEY_SYMBOLIC_STRUCTURE_GUARD, KEY_SYMBOLIC_TRUTH_CONTRACT_RECOVERY,
    KEY_TASK_FITTED_MODEL_REF, KEY_TASK_METRICS, KEY_TEXT_TOKEN_IDS,
    KEY_TIME_SERIES_DECOMPOSITION, KEY_TIME_SERIES_HORIZON, KEY_TIME_SERIES_MIN_TRAIN_SIZE,
    KEY_TIME_SERIES_OBJECTIVE_METRICS, KEY_TIME_SERIES_SEARCH_SPACE,
    KEY_TIME_SERIES_VALIDATION_SIZE, KEY_TIME_SERIES_WINDOW_CONFIG, KEY_TOKENIZER_VOCAB,
    KEY_TRAINER_CONTEXT, KEY_TRAINER_GET_STATE, KEY_TRAINER_REPORT,
    KEY_TRAINER_SNAPSHOT_STORE, KEY_TRAINER_STEP, KEY_TRAINING_RESULT, KEY_TRAINING_TASK,
    
    # Metrics
    KEY_METRICS, KEY_METRICS_MC_SAMPLES, KEY_METRICS_MC_MEAN, KEY_METRICS_MC_STD,
    KEY_METRICS_MC_MIN, KEY_METRICS_MC_MAX, KEY_METRICS_SURROGATE_STD,
    KEY_METRICS_IMPLICIT_RESIDUAL, KEY_METRICS_IMPLICIT_ITERS,
    KEY_METRICS_IMPLICIT_SUCCESS, KEY_METRICS_INNER_ELAPSED_MS, KEY_METRICS_INNER_STATUS,
    KEY_METRICS_INNER_CALLS, KEY_METRICS_SOFT_ERROR_COUNT, KEY_METRICS_SOFT_ERROR_LAST,
    
    # Context meta
    KEY_EVALUATION_COUNT, KEY_PARETO_SOLUTIONS, KEY_PARETO_OBJECTIVES, KEY_MUTATION_RATE,
    KEY_CROSSOVER_RATE, KEY_CONTEXT_SCHEMA, KEY_CONTEXT_EVENTS, KEY_CONTEXT_CACHE,
    KEY_DECISION_TRACE, KEY_CHECKPOINT_LATEST_PATH, KEY_CHECKPOINT_LAST_LOADED_PATH,
    KEY_BIAS_CACHE_FINGERPRINT, KEY_SNAPSHOT_KEY, KEY_SNAPSHOT_BACKEND,
    KEY_SNAPSHOT_SCHEMA, KEY_SNAPSHOT_META, KEY_BEST_CANDIDATE_REF, KEY_POPULATION_REF, KEY_OBJECTIVES_REF,
    KEY_CONSTRAINT_VIOLATIONS_REF, KEY_PARETO_SOLUTIONS_REF, KEY_PARETO_OBJECTIVES_REF,
    KEY_HISTORY_REF, KEY_DECISION_TRACE_REF, KEY_SEQUENCE_GRAPH_REF,
    KEY_BACKEND_WARM_START_REF, KEY_BACKEND_SOLUTION_POOL_REF, KEY_BACKEND_DIAGNOSTIC_REF,
    KEY_BACKEND_CALLBACK_REF,
}

CONTEXT_KEY_SET = frozenset(CANONICAL_CONTEXT_KEYS)
_CANONICAL_CONTEXT_KEY_BY_CASEFOLD = {
    key.casefold(): key for key in CANONICAL_CONTEXT_KEYS
}

# Standard metrics registry
METRIC_KEYS: tuple[str, ...] = (
    "objective",
    "loss",
    "accuracy",
    "auc_roc",
    "average_precision",
    "f1",
    "f1_macro",
    "mse",
    "rmse",
    "mae",
    "r2",
)

METRIC_FALLBACKS: tuple[str, ...] = ("strict", "safe_zero", "nan", "skip")


def normalize_context_key(key: str) -> str:
    """Normalize casing and whitespace without guessing another field name."""
    text = str(key).strip()
    if not text:
        return ""
    lowered = text.casefold()
    return _CANONICAL_CONTEXT_KEY_BY_CASEFOLD.get(lowered, lowered)


def normalize_context_keys(keys: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize a collection of context keys."""
    seen: set[str] = set()
    out: list[str] = []
    for key in tuple(keys or ()):
        normalized = normalize_context_key(str(key))
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return tuple(out)


def unknown_context_keys(keys: Iterable[str] | None) -> tuple[str, ...]:
    """Find unknown context keys."""
    return tuple(key for key in normalize_context_keys(keys) if key not in CONTEXT_KEY_SET)


def validate_context_keys(keys: Iterable[str] | None, *, strict: bool = False) -> tuple[str, ...]:
    """Validate context keys against the registry."""
    unknown = unknown_context_keys(keys)
    if strict and unknown:
        raise ValueError(f"unknown context keys: {unknown}")
    return unknown


def register_context_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """Return a deterministic merged registry tuple for external tooling."""
    return tuple(sorted({*CONTEXT_KEY_SET, *(normalize_context_key(key) for key in keys)}))
