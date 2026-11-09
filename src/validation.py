import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema

# Disable Pandera pandas import deprecation warning
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

ALL_CATEGORIES = [
    "ARTS", "ARTS & CULTURE", "BLACK VOICES", "BUSINESS", "COLLEGE", "COMEDY",
    "CRIME", "CULTURE & ARTS", "DIVORCE", "EDUCATION", "ENTERTAINMENT", "ENVIRONMENT",
    "FIFTY", "FOOD & DRINK", "GOOD NEWS", "GREEN", "HEALTHY LIVING", "HOME & LIVING",
    "IMPACT", "LATINO VOICES", "MEDIA", "MONEY", "PARENTING", "PARENTS", "POLITICS",
    "QUEER VOICES", "RELIGION", "SCIENCE", "SPORTS", "STYLE", "STYLE & BEAUTY",
    "TASTE", "TECH", "THE WORLDPOST", "TRAVEL", "U.S. NEWS", "WEDDINGS", "WEIRD NEWS",
    "WELLNESS", "WOMEN", "WORLD NEWS", "WORLDPOST"
]


def get_news_schema(strict: bool = False) -> DataFrameSchema:
    return DataFrameSchema(
        columns={
            "headline": Column(
                pa.String,
                checks=[
                    Check(lambda s: s.str.strip().str.len() >= 3, name="min_headline_len", error="Headline must be at least 3 characters.")
                ],
                nullable=False,
                coerce=False
            ),
            "category": Column(
                pa.String,
                checks=[
                    Check.isin(ALL_CATEGORIES, error="Invalid category: not in 42 valid news categories.")
                ],
                nullable=False,
                coerce=False
            ),
            "short_description": Column(
                pa.String,
                nullable=True,
                coerce=False
            ),
            "authors": Column(
                pa.String,
                nullable=True,
                coerce=False
            ),
            "date": Column(
                pa.String,
                checks=[
                    Check.str_matches(r"^\d{4}-\d{2}-\d{2}$", error="Date must conform to YYYY-MM-DD.")
                ],
                nullable=False,
                coerce=False
            ),
            "link": Column(
                pa.String,
                checks=[
                    Check.str_matches(r"^https?://.+", error="Link must be a valid HTTP/HTTPS URL.")
                ],
                nullable=False,
                coerce=False
            )
        },
        strict=strict,
        ordered=False
    )


def validate_schema(df: pd.DataFrame, strict: bool = False, output_report_name: str = "schema_validation_errors.csv"):
    schema = get_news_schema(strict=strict)
    try:
        schema.validate(df, lazy=True)
        return True, pd.DataFrame()
    except pa.errors.SchemaErrors as err:
        failures = err.failure_cases
        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        return False, failures
