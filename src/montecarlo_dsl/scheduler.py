from __future__ import annotations

from dataclasses import dataclass
import os
from itertools import islice
from operator import index as integer_index
from typing import Any, Callable, Iterator

from .cost_model import (
    ASTCostFeatures,
    CalibratedCostEstimate,
    InterpretableASTCostModel,
)


VALID_SCHEDULER_MODES = ("fixed", "ast_only", "pilot_only", "hybrid", "adaptive")


def normalize_scheduler_mode(mode: str) -> str:
    if mode not in VALID_SCHEDULER_MODES:
        raise ValueError(
            f"scheduler_mode must be one of {VALID_SCHEDULER_MODES}, got {mode!r}"
        )
    return "hybrid" if mode == "adaptive" else mode


def iter_runtime_schedule_tasks(
    *,
    total_iterations,
    logical_block_size,
    start_logical_block,
    initial_task_index,
    blocks_per_task,
    effective_workers,
):
    """Yield physical task tuples without allocating a task list."""
    block_size = max(1, int(logical_block_size))
    total_blocks = (int(total_iterations) + block_size - 1) // block_size
    block_index = max(0, int(start_logical_block))
    task_index = int(initial_task_index)
    blocks_per_task = max(1, int(blocks_per_task))
    workers = max(1, int(effective_workers))
    tail_threshold = (
        max(
            block_index,
            total_blocks - 2 * workers * max(1, blocks_per_task // 2),
        )
        if workers > 1 and blocks_per_task > 1
        else total_blocks
    )
    while block_index < total_blocks:
        step = (
            max(1, blocks_per_task // 2)
            if block_index >= tail_threshold and workers > 1
            else blocks_per_task
        )
        next_block = min(total_blocks, block_index + step)
        start_sample = block_index * block_size
        end_sample = min(int(total_iterations), next_block * block_size)
        yield (task_index, end_sample - start_sample, start_sample, end_sample)
        block_index = next_block
        task_index += 1


def count_runtime_schedule_tasks(
    *,
    total_iterations,
    logical_block_size,
    start_logical_block,
    blocks_per_task,
    effective_workers,
):
    """Count physical tasks in O(1) space and O(1) time."""
    block_size = max(1, int(logical_block_size))
    total_blocks = (int(total_iterations) + block_size - 1) // block_size
    start_block = max(0, int(start_logical_block))
    if start_block >= total_blocks:
        return 0
    blocks_per_task = max(1, int(blocks_per_task))
    workers = max(1, int(effective_workers))
    if workers > 1 and blocks_per_task > 1:
        tail_threshold = max(
            start_block,
            total_blocks - 2 * workers * max(1, blocks_per_task // 2),
        )
    else:
        tail_threshold = total_blocks
    pre_tail_blocks = max(0, tail_threshold - start_block)
    pre_tail_tasks = (pre_tail_blocks + blocks_per_task - 1) // blocks_per_task
    post_tail_start = min(total_blocks, start_block + pre_tail_tasks * blocks_per_task)
    if post_tail_start >= total_blocks:
        post_tail_tasks = 0
    else:
        tail_step = max(1, blocks_per_task // 2)
        post_tail_blocks = total_blocks - post_tail_start
        post_tail_tasks = (post_tail_blocks + tail_step - 1) // tail_step
    return int(pre_tail_tasks + post_tail_tasks)


def build_runtime_schedule_plan(
    *,
    total_iterations,
    batch_size,
    logical_block_size,
    max_workers,
    scheduler_mode,
    feature_values,
    static_per_sample_seconds,
    prior_equivalent_samples,
    serial_task_overhead_seconds,
    parallel_pool_spawn_seconds,
    parallel_task_ipc_seconds,
    estimated_bytes_per_sample,
    pilot_iterations=0,
    pilot_elapsed_seconds=None,
    memory_budget_bytes=32 * 1024 * 1024,
    parallel_speedup_margin=1.12,
    available_cpus=None,
    start_logical_block=0,
    initial_task_index=0,
):
    """Choose a schedule and expose its physical tasks as a lazy iterator.

    The returned ``tasks`` iterator is one-shot. ``task_generation`` contains the
    constant-size parameters needed to create a fresh iterator, and ``task_count``
    is calculated without enumerating the tasks.
    """
    mode = "hybrid" if scheduler_mode == "adaptive" else scheduler_mode
    start_logical_block = max(0, int(start_logical_block))
    initial_task_index = int(initial_task_index)
    block_size = min(max(1, int(logical_block_size)), max(1, int(total_iterations)))
    total_blocks = (int(total_iterations) + block_size - 1) // block_size
    remaining_blocks = max(0, total_blocks - int(start_logical_block))
    remaining_iterations = max(0, int(total_iterations) - int(start_logical_block) * block_size)
    static_cost = max(5.0e-9, float(static_per_sample_seconds))

    pilot_cost = None
    if mode == "ast_only" or pilot_elapsed_seconds is None or int(pilot_iterations) <= 0:
        effective_cost = static_cost
    else:
        branch_count = int(feature_values.get("branch_count", 1))
        block_overhead = float(serial_task_overhead_seconds) * (1.6 if branch_count == 2 else 1.0)
        net_pilot = max(1.0e-6, float(pilot_elapsed_seconds) - block_overhead)
        pilot_cost = max(5.0e-9, net_pilot / float(pilot_iterations))
        if mode == "pilot_only":
            effective_cost = pilot_cost
        else:
            prior_n = max(1, int(prior_equivalent_samples))
            alpha = min(0.88, max(0.15, float(pilot_iterations) / float(pilot_iterations + prior_n)))
            effective_cost = (1.0 - alpha) * static_cost + alpha * pilot_cost

    if remaining_blocks == 0:
        task_generation = {
            "total_iterations": int(total_iterations),
            "logical_block_size": block_size,
            "start_logical_block": int(start_logical_block),
            "initial_task_index": int(initial_task_index),
            "blocks_per_task": 1,
            "effective_workers": 1,
        }
        return {
            "tasks": iter_runtime_schedule_tasks(**task_generation),
            "task_generation": task_generation,
            "task_count": 0,
            "effective_workers": 1,
            "use_process_pool": False,
            "pilot_per_sample_seconds": pilot_cost,
            "effective_per_sample_seconds": effective_cost,
            "blocks_per_task": 1,
            "estimated_serial_seconds": 0.0,
            "estimated_selected_seconds": 0.0,
            "selection_reason": "completed_in_pilot_block",
            "logical_block_size": block_size,
            "logical_block_count": total_blocks,
        }

    cpu_cap = int(available_cpus or os.cpu_count() or 1)
    max_viable_workers = min(max(1, int(max_workers)), cpu_cap, remaining_blocks)
    max_samples_per_task = max(
        block_size,
        int(memory_budget_bytes) // max(8, int(estimated_bytes_per_sample)),
    )
    max_blocks_by_memory = max(1, min(remaining_blocks, max_samples_per_task // block_size))

    target_serial_tasks = min(remaining_blocks, 4 if remaining_iterations <= 250_000 else 8)
    serial_blocks_per_task = min(
        max_blocks_by_memory,
        max(1, (remaining_blocks + max(1, target_serial_tasks) - 1) // max(1, target_serial_tasks)),
    )
    serial_task_count = (remaining_blocks + serial_blocks_per_task - 1) // serial_blocks_per_task
    pure_work_seconds = float(remaining_iterations) * effective_cost
    estimated_serial_seconds = (
        pure_work_seconds + serial_task_count * float(serial_task_overhead_seconds)
    )

    best_workers = 1
    best_blocks_per_task = serial_blocks_per_task
    best_seconds = estimated_serial_seconds
    best_reason = "serial_in_process_dominates_coordination_overhead"

    for workers in range(2, max_viable_workers + 1):
        target_parallel_tasks = min(
            remaining_blocks,
            max(2 * workers, min(4 * workers, remaining_blocks)),
        )
        candidate_blocks = min(
            max_blocks_by_memory,
            max(1, remaining_blocks // target_parallel_tasks),
        )
        task_count = (remaining_blocks + candidate_blocks - 1) // candidate_blocks
        efficiency = max(0.55, 1.0 - 0.08 * (workers - 1))
        imbalance_tail = 0.25 * (candidate_blocks * block_size * effective_cost)
        candidate_seconds = (
            float(parallel_pool_spawn_seconds)
            + pure_work_seconds / (workers * efficiency)
            + task_count * float(parallel_task_ipc_seconds)
            + imbalance_tail
        )
        if candidate_seconds * float(parallel_speedup_margin) < best_seconds:
            best_workers = workers
            best_blocks_per_task = candidate_blocks
            best_seconds = candidate_seconds
            best_reason = f"parallel_w{workers}_amortizes_pool_and_ipc_cost"

    task_generation = {
        "total_iterations": int(total_iterations),
        "logical_block_size": block_size,
        "start_logical_block": int(start_logical_block),
        "initial_task_index": int(initial_task_index),
        "blocks_per_task": int(best_blocks_per_task),
        "effective_workers": int(best_workers),
    }
    task_count = count_runtime_schedule_tasks(
        total_iterations=task_generation["total_iterations"],
        logical_block_size=task_generation["logical_block_size"],
        start_logical_block=task_generation["start_logical_block"],
        blocks_per_task=task_generation["blocks_per_task"],
        effective_workers=task_generation["effective_workers"],
    )
    return {
        "tasks": iter_runtime_schedule_tasks(**task_generation),
        "task_generation": task_generation,
        "task_count": task_count,
        "effective_workers": best_workers,
        "use_process_pool": best_workers > 1,
        "pilot_per_sample_seconds": pilot_cost,
        "effective_per_sample_seconds": effective_cost,
        "blocks_per_task": best_blocks_per_task,
        "estimated_serial_seconds": estimated_serial_seconds,
        "estimated_selected_seconds": best_seconds,
        "selection_reason": best_reason,
        "logical_block_size": block_size,
        "logical_block_count": total_blocks,
    }


@dataclass(frozen=True, slots=True)
class PhysicalTaskSpec:
    """Especificacion de una tarea fisica que agrupa uno o mas bloques logicos."""

    task_index: int
    start_sample: int
    end_sample: int
    start_logical_block: int
    end_logical_block: int

    @property
    def iterations(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def logical_block_count(self) -> int:
        return self.end_logical_block - self.start_logical_block

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.task_index, self.iterations, self.start_sample, self.end_sample)


@dataclass(frozen=True, slots=True)
class LazyPhysicalTaskSequence:
    """Re-iterable physical-task sequence with O(1) descriptor storage."""

    _factory: Callable[[], Iterator[PhysicalTaskSpec]]
    _count: int

    def __iter__(self) -> Iterator[PhysicalTaskSpec]:
        return self._factory()

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        """Support tuple-like indexing without materializing the full task stream."""
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            if step < 0:
                # Reverse slicing is an explicit materialization request.
                return tuple(self)[index]
            return tuple(islice(self._factory(), start, stop, step))
        normalized = integer_index(index)
        if normalized < 0:
            normalized += self._count
        if normalized < 0 or normalized >= self._count:
            raise IndexError("physical task index out of range")
        return next(islice(self._factory(), normalized, normalized + 1))


def iter_fixed_physical_tasks(
    *, total_iterations: int, batch_size: int, logical_block_size: int
) -> Iterator[PhysicalTaskSpec]:
    """Yield fixed-size physical tasks without constructing a tuple/list."""
    if batch_size <= 0 or logical_block_size <= 0:
        raise ValueError("batch_size and logical_block_size must be positive")
    offset = 0
    task_index = 0
    while offset < total_iterations:
        end = min(total_iterations, offset + batch_size)
        start_block = offset // logical_block_size
        end_block = (end - 1) // logical_block_size + 1
        yield PhysicalTaskSpec(
            task_index=task_index,
            start_sample=offset,
            end_sample=end,
            start_logical_block=start_block,
            end_logical_block=end_block,
        )
        offset = end
        task_index += 1


@dataclass(frozen=True, slots=True)
class ExecutionSchedulePlan:
    """Decision de planificacion seleccionada por el planificador consciente del coste."""

    scheduler_mode: str
    total_iterations: int
    logical_block_size: int
    logical_block_count: int
    effective_workers: int
    requested_max_workers: int
    use_process_pool: bool
    blocks_per_task: int
    physical_task_size: int
    tasks: LazyPhysicalTaskSequence
    cost_estimate: CalibratedCostEstimate
    estimated_serial_seconds: float
    estimated_selected_seconds: float
    selection_reason: str

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    def task_tuples(self) -> list[tuple[int, int, int, int]]:
        """Materialize task tuples explicitly; core compilation/runtime paths stay lazy."""
        return [task.to_tuple() for task in self.tasks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler_mode": self.scheduler_mode,
            "total_iterations": self.total_iterations,
            "logical_block_size": self.logical_block_size,
            "logical_block_count": self.logical_block_count,
            "effective_workers": self.effective_workers,
            "requested_max_workers": self.requested_max_workers,
            "use_process_pool": self.use_process_pool,
            "blocks_per_task": self.blocks_per_task,
            "physical_task_size": self.physical_task_size,
            "task_count": self.task_count,
            "estimated_serial_seconds": round(self.estimated_serial_seconds, 6),
            "estimated_selected_seconds": round(self.estimated_selected_seconds, 6),
            "selection_reason": self.selection_reason,
            "cost_estimate": self.cost_estimate.to_dict(),
        }


class AdaptiveMonteCarloScheduler:
    """Planificador adaptativo guiado por el AST y calibracion piloto (Componente B).

    Separa estrictamente:
    - Bloque logico (``logical_block_size``): unidad canonica de generacion RNG y
      reduccion determinista de estadisticos.
    - Tarea fisica (``PhysicalTaskSpec``): agrupacion de bloques logicos asignada
      a ejecucion serial en proceso o a un worker de ``ProcessPoolExecutor``.
    """

    def __init__(
        self,
        cost_model: InterpretableASTCostModel | None = None,
        *,
        memory_budget_bytes: int = 32 * 1024 * 1024,
        parallel_speedup_margin: float = 1.12,
    ) -> None:
        self.cost_model = cost_model or InterpretableASTCostModel()
        self.memory_budget_bytes = memory_budget_bytes
        self.parallel_speedup_margin = parallel_speedup_margin

    def build_plan(
        self,
        features: ASTCostFeatures,
        *,
        total_iterations: int,
        batch_size: int = 25_000,
        logical_block_size: int = 5_000,
        max_workers: int = 4,
        scheduler_mode: str = "hybrid",
        pilot_iterations: int = 0,
        pilot_elapsed_seconds: float | None = None,
        available_cpus: int | None = None,
        start_logical_block: int = 0,
        initial_task_index: int = 0,
    ) -> ExecutionSchedulePlan:
        mode = normalize_scheduler_mode(scheduler_mode)
        eff_block_size = min(max(1, logical_block_size), max(1, total_iterations))
        total_logical_blocks = (total_iterations + eff_block_size - 1) // eff_block_size
        remaining_blocks = max(0, total_logical_blocks - start_logical_block)
        remaining_iterations = max(0, total_iterations - start_logical_block * eff_block_size)

        if mode == "fixed":
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            fixed_task_count = (total_iterations + batch_size - 1) // batch_size
            tasks = LazyPhysicalTaskSequence(
                _factory=lambda: iter_fixed_physical_tasks(
                    total_iterations=total_iterations,
                    batch_size=batch_size,
                    logical_block_size=eff_block_size,
                ),
                _count=fixed_task_count,
            )
            workers = min(max_workers, max(1, fixed_task_count))
            use_pool = (workers > 1) or (total_iterations >= logical_block_size)
            est = self.cost_model.calibrate_with_pilot(features, mode="ast_only")
            est_serial = self.cost_model.estimate_total_seconds(
                features, total_iterations, num_tasks=fixed_task_count, workers=1
            )
            est_sel = self.cost_model.estimate_total_seconds(
                features, total_iterations, num_tasks=fixed_task_count, workers=workers
            )
            return ExecutionSchedulePlan(
                scheduler_mode="fixed",
                total_iterations=total_iterations,
                logical_block_size=eff_block_size,
                logical_block_count=total_logical_blocks,
                effective_workers=workers,
                requested_max_workers=max_workers,
                use_process_pool=use_pool,
                blocks_per_task=max(1, batch_size // eff_block_size),
                physical_task_size=batch_size,
                tasks=tasks,
                cost_estimate=est,
                estimated_serial_seconds=est_serial,
                estimated_selected_seconds=est_sel,
                selection_reason="fixed_batch_and_worker_configuration",
            )

        cost_estimate = self.cost_model.calibrate_with_pilot(
            features,
            pilot_iterations=pilot_iterations,
            pilot_elapsed_seconds=pilot_elapsed_seconds,
            mode=mode,
        )
        runtime_plan = build_runtime_schedule_plan(
            total_iterations=total_iterations,
            batch_size=batch_size,
            logical_block_size=eff_block_size,
            max_workers=max_workers,
            scheduler_mode=mode,
            feature_values=features.to_dict(),
            static_per_sample_seconds=cost_estimate.static_per_sample_seconds,
            prior_equivalent_samples=self.cost_model.prior_equivalent_samples,
            serial_task_overhead_seconds=self.cost_model.serial_task_overhead_seconds,
            parallel_pool_spawn_seconds=self.cost_model.parallel_pool_spawn_seconds,
            parallel_task_ipc_seconds=self.cost_model.parallel_task_ipc_seconds,
            estimated_bytes_per_sample=features.estimated_bytes_per_sample,
            pilot_iterations=pilot_iterations,
            pilot_elapsed_seconds=pilot_elapsed_seconds,
            memory_budget_bytes=self.memory_budget_bytes,
            parallel_speedup_margin=self.parallel_speedup_margin,
            available_cpus=available_cpus,
            start_logical_block=start_logical_block,
            initial_task_index=initial_task_index,
        )
        task_generation = dict(runtime_plan["task_generation"])

        def _task_factory():
            for task in iter_runtime_schedule_tasks(**task_generation):
                yield PhysicalTaskSpec(
                    task_index=int(task[0]),
                    start_sample=int(task[2]),
                    end_sample=int(task[3]),
                    start_logical_block=int(task[2] // eff_block_size),
                    end_logical_block=int((task[3] + eff_block_size - 1) // eff_block_size),
                )

        tasks = LazyPhysicalTaskSequence(
            _factory=_task_factory,
            _count=int(runtime_plan["task_count"]),
        )
        return ExecutionSchedulePlan(
            scheduler_mode=mode,
            total_iterations=total_iterations,
            logical_block_size=eff_block_size,
            logical_block_count=total_logical_blocks,
            effective_workers=runtime_plan["effective_workers"],
            requested_max_workers=max_workers,
            use_process_pool=runtime_plan["use_process_pool"],
            blocks_per_task=runtime_plan["blocks_per_task"],
            physical_task_size=runtime_plan["blocks_per_task"] * eff_block_size,
            tasks=tasks,
            cost_estimate=cost_estimate,
            estimated_serial_seconds=runtime_plan["estimated_serial_seconds"],
            estimated_selected_seconds=runtime_plan["estimated_selected_seconds"],
            selection_reason=runtime_plan["selection_reason"],
        )

    @staticmethod
    def _build_fixed_tasks(
        *,
        total_iterations: int,
        batch_size: int,
        logical_block_size: int,
    ) -> tuple[PhysicalTaskSpec, ...]:
        """Compatibility helper; callers explicitly requesting a tuple opt into materialization."""
        return tuple(iter_fixed_physical_tasks(
            total_iterations=total_iterations,
            batch_size=batch_size,
            logical_block_size=logical_block_size,
        ))

    @staticmethod
    def _build_grouped_logical_tasks(
        *,
        total_iterations: int,
        logical_block_size: int,
        total_logical_blocks: int,
        start_logical_block: int,
        blocks_per_task: int,
        workers: int,
        initial_task_index: int = 0,
    ) -> tuple[PhysicalTaskSpec, ...]:
        tasks: list[PhysicalTaskSpec] = []
        b = start_logical_block
        task_index = initial_task_index

        tail_threshold = (
            max(start_logical_block, total_logical_blocks - 2 * workers * max(1, blocks_per_task // 2))
            if workers > 1 and blocks_per_task > 1
            else total_logical_blocks
        )

        while b < total_logical_blocks:
            step = (
                max(1, blocks_per_task // 2)
                if b >= tail_threshold and workers > 1
                else blocks_per_task
            )
            next_b = min(total_logical_blocks, b + step)
            start_sample = b * logical_block_size
            end_sample = min(total_iterations, next_b * logical_block_size)
            tasks.append(
                PhysicalTaskSpec(
                    task_index=task_index,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_logical_block=b,
                    end_logical_block=next_b,
                )
            )
            b = next_b
            task_index += 1

        return tuple(tasks)
