import dataset
import unittest


class TestDiv2(unittest.TestCase):
    def test_div(self):
        self.assertEqual(dataset.max_2_div(80), 16)


if __name__ == "__main__":
    unittest.main()
