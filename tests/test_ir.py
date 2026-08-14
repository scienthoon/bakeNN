import unittest

import numpy as np

from bakenn import FloatLinear, FloatMLP, quantize_ptq
from bakenn.plan import lower_to_plan


class IrAndPlannerTests(unittest.TestCase):
    def test_linear_chain_has_static_reused_arena(self):
        rng = np.random.default_rng(10)
        model = FloatMLP(
            (
                FloatLinear(rng.normal(size=(7, 8)), rng.normal(size=7), True, "one"),
                FloatLinear(rng.normal(size=(5, 7)), rng.normal(size=5), True, "two"),
                FloatLinear(rng.normal(size=(6, 5)), rng.normal(size=6), True, "three"),
                FloatLinear(rng.normal(size=(3, 6)), rng.normal(size=3), False, "four"),
            ),
            "tiny_mlp",
        )
        graph = quantize_ptq(model, rng.normal(size=(64, 8)))
        plan = lower_to_plan(graph)

        self.assertEqual(plan.arithmetic_profile, "bakenn.int8.v1")
        self.assertEqual(len(plan.steps), 4)
        self.assertEqual(plan.arena_size % plan.arena_alignment, 0)
        self.assertGreater(plan.arena_size, 0)
        self.assertEqual(plan.tensors["one.output"].offset, plan.tensors["three.output"].offset)
        self.assertTrue(all(max(step.accumulator_bounds) < 2**31 for step in plan.steps))


if __name__ == "__main__":
    unittest.main()
