import os
import unittest
import pandas as pd
from pathlib import Path


class MyTestCase(unittest.TestCase):
    def test_read_file(self):
        data = pd.read_csv("F:\\科研类\\codes\\hydro-routing-cnn\\example\\data\\sanxia\\1-pingshan-1950-2007-day-runoff.csv",
                           encoding="UTF-8")
        print(data)

    def test_split_str(self):
        str1 = '10月11日'
        print(int(str1.split('月')[0]))
        print(int(str1.split('月')[1][:-1]))


if __name__ == '__main__':
    unittest.main()
