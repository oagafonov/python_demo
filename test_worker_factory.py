from unittest import TestCase

from analytics.domain import Prediction, Box, Point
from analytics.domain.worker_factory import WorkerFactory


class TestWorkerFactory(TestCase):
    def test_worker_without_intersection(self):
        predictions = [Prediction(label='worker',
                                  confidence=0.9996539354324341,
                                  box=Box(bottom_left=Point(1145, 744),
                                          top_right=Point(1456, 997))),
                       Prediction(label='worker',
                                  confidence=0.9994698166847229,
                                  box=Box(bottom_left=Point(923, 501),
                                          top_right=Point(1107, 880))),
                       Prediction(label='worker',
                                  confidence=0.998865008354187,
                                  box=Box(bottom_left=Point(1485, 912),
                                          top_right=Point(1709, 1067))),
                       Prediction(label='head_in_hh',
                                  confidence=0.9984935522079468,
                                  box=Box(bottom_left=Point(991, 499),
                                          top_right=Point(1066, 591))),
                       Prediction(label='head_in_hh',
                                  confidence=0.9975487589836121,
                                  box=Box(bottom_left=Point(1291, 746),
                                          top_right=Point(1375, 849))),
                       Prediction(label='head_in_hh',
                                  confidence=0.9916406869888306,
                                  box=Box(bottom_left=Point(1625, 921),
                                          top_right=Point(1708, 1005))),
                       ]
        factory = WorkerFactory(predictions)
        workers = list(factory.get_workers())
        assert len(workers) == 3

        worker = workers[0]
        assert worker.confidence == min(1.0, 0.9996539354324341 * (1 + 0.09))
        assert worker.box == predictions[0].box
        assert worker.head.has_helmet
        assert worker.head.box == predictions[4].box
        assert worker.head.confidence == predictions[4].confidence

        worker = workers[1]
        assert worker.confidence == min(1.0, 0.9994698166847229 * (1 + 0.09))
        assert worker.box == predictions[1].box
        assert worker.head.has_helmet
        assert worker.head.box == predictions[3].box
        assert worker.head.confidence == predictions[3].confidence

        worker = workers[2]
        assert worker.confidence == min(1.0, 0.998865008354187 * (1 + 0.09))
        assert worker.box == predictions[2].box
        assert worker.head.has_helmet
        assert worker.head.box == predictions[5].box
        assert worker.head.confidence == predictions[5].confidence

    def test_workers_with_intersection(self):
        predictions = [Prediction(label='worker',
                                  confidence=0.9984087347984314,
                                  box=Box(bottom_left=Point(842, 34),
                                          top_right=Point(1143, 537))),
                       Prediction(label='worker',
                                  confidence=0.9951882362365723,
                                  box=Box(bottom_left=Point(938, 313),
                                          top_right=Point(1331, 760))),
                       Prediction(label='head_in_hh',
                                  confidence=0.8822525143623352,
                                  box=Box(bottom_left=Point(842, 33),
                                          top_right=Point(1005, 201))),
                       ]
        factory = WorkerFactory(predictions)
        workers = list(factory.get_workers())
        assert len(workers) == 2

        worker = workers[1]
        assert worker.confidence == min(1.0, 0.9984087347984314 * (1 - 0.09))
        assert worker.box == predictions[0].box
        assert worker.head is None

        worker = workers[0]
        assert worker.confidence == min(1.0, 0.9951882362365723 * (1 - 0.09))
        assert worker.box == predictions[1].box
        assert worker.head is None

    def test_when_worker_with_small_head_then_reduce_confidence(self):
        predictions = [
            Prediction(label='worker',
                       confidence=0.9016635417938232,
                       box=Box(bottom_left=Point(746, 103),
                               top_right=Point(1025, 683))),
            Prediction(label='head_in_hh',
                       confidence=0.9662376642227173,
                       box=Box(bottom_left=Point(794, 244),
                               top_right=Point(872, 323))),

        ]
        factory = WorkerFactory(predictions)
        workers = list(factory.get_workers())
        assert len(workers) == 0

    def test_when_worker_many_heads(self):
        predictions = [
            Prediction(label='worker',
                       confidence=0.981,
                       box=Box(bottom_left=Point(3, 21),
                               top_right=Point(8, 37))),
            Prediction(label='worker',
                       confidence=0.972,
                       box=Box(bottom_left=Point(4, 16),
                               top_right=Point(8, 41))),
            Prediction(label='worker',
                       confidence=0.983,
                       box=Box(bottom_left=Point(6, 25),
                               top_right=Point(8, 40))),
            Prediction(label='worker',
                       confidence=0.984,
                       box=Box(bottom_left=Point(1, 1),
                               top_right=Point(5, 24))),
            Prediction(label='worker',
                       confidence=0.985,
                       box=Box(bottom_left=Point(4, 1),
                               top_right=Point(9, 21))),
            Prediction(label='head_in_hh',
                       confidence=0.966,
                       box=Box(bottom_left=Point(3, 14),
                               top_right=Point(5, 26))),
            Prediction(label='head_in_hh',
                       confidence=0.967,
                       box=Box(bottom_left=Point(3, 29),
                               top_right=Point(8, 33))),
            Prediction(label='head_in_hh',
                       confidence=0.968,
                       box=Box(bottom_left=Point(6, 12),
                               top_right=Point(8, 23))),

        ]
        factory = WorkerFactory(predictions)
        workers = list(factory.get_workers())

        assert len(workers) == 3

        worker = workers[0]
        assert worker.box == predictions[4].box
        assert worker.head.box == predictions[7].box

        worker = workers[1]
        assert worker.box == predictions[1].box
        assert worker.head.box == predictions[6].box

        worker = workers[2]
        assert worker.box == predictions[3].box
        assert worker.head.box == predictions[5].box
