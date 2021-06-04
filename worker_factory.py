import copy
import functools
from typing import List, Optional, Iterable, Tuple

from .model import Prediction, Worker, Head


def is_worker(prediction: Prediction) -> bool:
    return prediction.label == 'worker' and prediction.confident


def is_helmet(prediction: Prediction) -> bool:
    return prediction.label == 'head_in_hh' and prediction.confident


def is_head(prediction: Prediction) -> bool:
    return prediction.confident and (prediction.label == 'head_in_hh' or prediction.label == 'head_wout_hh')


def order_worker(worker: Worker) -> Tuple[float, float, float]:
    return worker.box.area, worker.confidence, worker.box.top_right.x


def order_workers(workers: Iterable[Worker]) -> Iterable[Worker]:
    return sorted(list(workers), key=order_worker, reverse=True)


def get_neighbors(worker: Worker, workers: Iterable[Worker]) -> Iterable[Worker]:
    neighbors = set()
    candidates = {candidate for candidate in workers if candidate != worker}

    for candidate in order_workers(candidates):
        if candidate.box.intersection_factor(worker.box) >= 0.8:
            neighbors.add(candidate)

    distant_candidates = order_workers(candidates.difference(neighbors))
    if distant_candidates:
        for neighbor in copy.copy(neighbors):
            distant_neighbors = get_neighbors(neighbor, distant_candidates)
            if distant_neighbors:
                neighbors = neighbors.union(distant_neighbors)

    return order_workers(neighbors)


def get_large_worker(worker1: Worker, worker2: Optional[Worker]) -> Worker:
    if worker2:
        if worker1.box.area > worker2.box.area:
            return worker1
        elif worker1.box.area == worker2.box.area:
            return worker1 if worker1.confidence >= worker2.confidence else worker2

    return worker1


class WorkerFactory:
    def __init__(self, predictions: List[Prediction]):
        self.predictions = predictions

    def get_workers(self) -> Iterable[Worker]:
        all_workers = order_workers([Worker(box=prediction.box,
                                            confidence=prediction.confidence,
                                            head=None) for prediction in self.predictions if is_worker(prediction)])
        heads = {Head(box=prediction.box,
                      confidence=prediction.confidence,
                      has_helmet=is_helmet(prediction)) for prediction in self.predictions if is_head(prediction)}

        workers = set()
        processed_workers = set()
        for worker in all_workers:
            if worker in processed_workers:
                continue

            candidates = {worker}
            neighbors = get_neighbors(worker, all_workers)

            if neighbors:
                candidates = candidates.union(neighbors)

            processed_workers = processed_workers.union(candidates)
            large_worker = functools.reduce(get_large_worker, order_workers(candidates))
            workers.add(large_worker)

        result = set()

        for worker in order_workers(workers):
            worker_heads = worker.get_fit_heads(heads)
            if len(worker_heads) == 1:
                worker_head = worker_heads[0]
                result.add(worker.head_found(worker_head))
                heads.remove(worker_head)
            elif len(worker_heads) == 0:
                result.add(worker.head_not_found())

        workers_with_many_heads = order_workers(workers.difference(result))

        for worker in workers_with_many_heads:
            worker_heads = worker.get_fit_heads(heads)
            if worker_heads:
                worker_head = worker_heads[0]
                result.add(worker.head_found(worker_head))
                heads.remove(worker_head)
            else:
                result.add(worker.head_not_found())

        return order_workers([worker for worker in result if worker.confident])
