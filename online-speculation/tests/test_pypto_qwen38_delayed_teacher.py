"""验证跨轮教师前缀，使用 token ID，不运行模型或 NPU。"""

import unittest

from blockspec.pypto_qwen38.delayed_teacher import DelayedTeacherBlock


class DelayedTeacherTests(unittest.TestCase):
    def test_future_rows_follow_the_correction_not_the_rejected_candidate(self):
        block = DelayedTeacherBlock("r", [10], width=6)
        # 首轮接受 11、12，将错误候选替换为 99。仅前三行教师分布有效。
        block.append_verified("r", [10], [10, 11, 12], [11, 12, 99], [[0], [1], [2]])
        self.assertFalse(block.ready)
        with self.assertRaises(ValueError):
            block.append_verified("r", [10, 11, 12, 99], [13], [15], [[3]])
        self.assertEqual(len(block.rows), 3)
        # 下一轮真正从 99 出发，补齐原 anchor 后的两行教师分布。
        block.append_verified("r", [10, 11, 12, 99], [99, 15], [15, 16], [[3], [4]])
        self.assertTrue(block.full)
        self.assertEqual(block.rows, [[0], [1], [2], [3], [4]])

    def test_a_short_request_does_not_invent_missing_targets(self):
        block = DelayedTeacherBlock("r", [10])
        block.append_verified("r", [10], [10], [11], [[0]])
        self.assertTrue(block.finish())
        self.assertFalse(block.full)
        self.assertEqual(len(block.rows), 1)

    def test_request_ids_and_corrected_prefix_are_isolated(self):
        block = DelayedTeacherBlock("r", [10])
        for request, prefix in (("other", [10]), ("r", [20])):
            with self.assertRaises(ValueError):
                block.append_verified(request, prefix, [prefix[-1]], [11], [[0]])
        self.assertEqual(block.rows, [])

    def test_snapshots_survive_output_buffer_reuse(self):
        block = DelayedTeacherBlock("r", [10], width=2)
        source = [[0.1, 0.9]]
        block.append_verified("r", [10], [10], [11], source)
        source[0][0] = 99
        self.assertEqual(block.rows, [[0.1, 0.9]])

    def test_empty_and_complete_windows_have_explicit_boundaries(self):
        block = DelayedTeacherBlock("r", [10], width=2)
        self.assertFalse(block.finish())
        with self.assertRaises(ValueError):
            block.append_verified("r", [10], [10], [11], [[0]])


if __name__ == "__main__":
    unittest.main()
