import unittest
from decimal import Decimal as D

from app.numparse import numbers_in


class NumbersIn(unittest.TestCase):
    def check(self, text, expected):
        self.assertEqual(numbers_in(text), [D(str(x)) for x in expected], text)

    def test_digits_with_scale(self):
        self.check("about R6.2 million", [6200000])
        self.check("R6.2m", [6200000])
        self.check("~R8.5m to invest", [8500000])
        self.check("R45k a month", [45000])
        self.check("USD 1.2m offshore", [1200000])

    def test_digits_with_separators(self):
        self.check("roughly R28,000 a month", [28000])
        self.check("R 6 200 000", [6200000])
        self.check("R4,500,000", [4500000])

    def test_percentages(self):
        self.check("Adviser fee 0.60%", [D("0.60")])
        self.check("fee 0.5%", [D("0.5")])
        self.assertEqual(numbers_in("0.60%")[0], D("0.6"))

    def test_spoken(self):
        self.check("call it four and a half million rand", [4500000])
        self.check("zero point seven five", [D("0.75")])
        self.check("let's say CPI plus five", [5])
        self.check("ten years plus", [10])
        self.check("they are both around fifty", [50])
        self.check("two hundred and fifty thousand", [250000])
        self.check("twenty five", [25])

    def test_mixed(self):
        self.check("CPI + 5%", [5])
        self.check("15+ years", [15])
        self.check("aged 63", [63])
        self.check("7-year horizon", [7])
        self.check("10-year+ view", [10])

    def test_not_numbers(self):
        self.check("a few million, rand", [])
        self.check("no one knows", [])
        self.check("the one thing", [])
        self.check("6 months", [6])  # 'months' must not be read as 'm'
        self.check("", [])
        self.check("balanced-ish feel", [])


if __name__ == "__main__":
    unittest.main()