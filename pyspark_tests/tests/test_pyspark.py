from __future__ import annotations
import pyspark
import unittest

from pyspark_tests.src.query_pyspark import get_family_data, get_spark_session, get_elder_child, get_initials_col, \
    get_initials

spark: pyspark.sql.SparkSession | None = None


def setUpModule():
    global spark
    spark = get_spark_session('local')


def tearDownModule():
    global spark
    if spark is None:
        return
    try:
        spark.stop()
    finally:
        spark = None


class TestPysparkQueries(unittest.TestCase):
    def test_session_created(self):
        self.assertIsNotNone(spark)

    def test_load_data(self):
        df = get_family_data()
        self.assertEqual(25, df.count())

    def test_elder_child_query(self):
        df = get_elder_child(get_family_data())
        elders = {_.elder_child for _ in df.toLocalIterator()}
        self.assertEqual(elders, {'Gus', 'Rita', 'Sam', 'Trent', 'Ursula'})

    def test_get_initials_col_1_by_1(self):
        df = (
            get_family_data()
            .withColumn('initials', get_initials_col())
            .orderBy('date_born')
        )
        expected_list = ['V. A.', 'W. W.', 'X. M.', 'Y. T.', 'Z. C.', 'I. M.', 'J. T.', 'K. C.', 'L. A.', 'M. W.',
                         'N. M.', 'O. T.', 'P. C.', 'Q. A.', 'A. A.', 'B. W.', 'C. M.', 'E. T.', 'F. C.', 'G. A.',
                         'H. W.', 'R. W.', 'S. M.', 'T. T.', 'U. C.']
        for expected, actual in zip(expected_list, [_.initials for _ in df.toLocalIterator()]):
            self.assertEqual(expected, actual)

    def test_get_initials(self):
        self.assertEqual('B. H.', get_initials('Bob', 'Hope'))
        self.assertEqual('C. C.', get_initials('Charlie', 'Chaplin'))
        self.assertEqual('J. L.', get_initials('Jonathan', 'Livingstone'))

    def test_get_initials_col_support_function(self):
        df = (
            get_family_data()
            .withColumn('initials', get_initials_col())
        )
        for row in df.toLocalIterator():
            self.assertEqual(get_initials(row.first_name, row.last_name), row.initials)


if __name__ == '__main__':
    unittest.main()
