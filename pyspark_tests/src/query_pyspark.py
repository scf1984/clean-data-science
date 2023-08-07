import os

import pyspark
from pyspark.sql import functions as f


def get_spark_session(scope='local'):
    if scope == 'local':
        return (
            pyspark.sql.SparkSession.builder
            .appName('unit-tests')
            .master('local[4]')
        ).getOrCreate()
    else:
        ...  # TODO


def get_family_data():
    return (
        get_spark_session(scope='local')
        .read.csv(os.path.join(os.path.dirname(__file__), '../assets/data_sample.csv'), header=True)
    )


def get_elder_child(family_df: pyspark.sql.DataFrame):
    return (
        family_df
        .orderBy(f.col('date_born').desc())
        .groupby('last_name')
        .agg(f.first('first_name').alias('elder_child'))
    )


def get_initials(first_name, last_name):
    return f'{first_name[:1]}. {last_name[:1]}.'


def get_initials_col():
    return (
        f.concat(
            f.substring('first_name', 0, 1),
            f.lit('. '),
            f.substring('last_name', 0, 1),
            f.lit('.'),
        )

    ).alias('initials')
