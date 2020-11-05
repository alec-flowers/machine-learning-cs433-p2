import unittest
import load

class TestLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('setUpClass')

    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def test_load_hrf(self):
        data = load.load_hrf()
        n_regions, n_timecourses, n_subjects  = data.shape

        self.assertEqual(n_subjects, 100)
        self.assertEqual(n_regions, 379)

        with self.assertRaises(AssertionError):
            load.load_hrf('hello')

    def test_load_task_paradigms(self):
        data = load.load_task_paradigms()
        n_subjects = len(data)

        self.assertEqual(n_subjects, 100)

        with self.assertRaises(AssertionError):
            load.load_task_paradigms('hello')

if __name__ == '__main__':
    unittest.main()
