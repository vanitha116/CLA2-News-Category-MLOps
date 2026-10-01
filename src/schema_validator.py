import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
from src.config import ALL_CATEGORIES, ARTIFACTS_DIR

def get_news_category_schema(strict: bool = True) -> DataFrameSchema:
    """
    Defines the strict schema specification for the News Category dataset using Pandera.
    Ensures datatype consistency, categorical validity, URL syntax, and date integrity.
    """
    return DataFrameSchema(
        columns={
            "headline": Column(
                pa.String,
                checks=[
                    Check(lambda s: s.str.strip().str.len() >= 3, name="min_headline_length_3", error="Headline must be at least 3 characters.")
                ],
                nullable=False,
                coerce=False,
                description="Primary headline text of the news article"
            ),
            "category": Column(
                pa.String,
                checks=[
                    Check.isin(ALL_CATEGORIES, error="Invalid category: must be one of the 42 official categories.")
                ],
                nullable=False,
                coerce=False,
                description="Categorical target label for classification"
            ),
            "short_description": Column(
                pa.String,
                nullable=True,
                coerce=False,
                description="Brief summary or description of the news article"
            ),
            "authors": Column(
                pa.String,
                nullable=True,
                coerce=False,
                description="Author name(s) credited for the article"
            ),
            "date": Column(
                pa.String,
                checks=[
                    Check.str_matches(r"^\d{4}-\d{2}-\d{2}$", error="Date must conform to YYYY-MM-DD ISO format.")
                ],
                nullable=False,
                coerce=False,
                description="Publication date string in ISO format"
            ),
            "link": Column(
                pa.String,
                checks=[
                    Check.str_matches(r"^https?://.+", error="Link must be a valid HTTP/HTTPS URL.")
                ],
                nullable=False,
                coerce=False,
                description="Web URL pointing to original HuffPost article"
            )
        },
        strict=strict,
        ordered=False
    )

def inject_synthetic_corruptions(clean_df: pd.DataFrame, num_corruptions: int = 5) -> pd.DataFrame:
    """
    Generates a synthetic corrupted dataset containing deliberate flaws:
    1. Datatype mismatch: Numeric value placed into string column
    2. Missing attributes: Null value injected into mandatory column
    3. Invalid categorical values: Unrecognized category label
    4. Corrupted records: Malformed date and invalid URL pattern
    5. Schema inconsistency: Extraneous unauthorized column
    """
    corrupted = clean_df.head(num_corruptions).copy()

    # Corruption 1: Datatype mismatch in 'headline' (float/int instead of string)
    corrupted.loc[0, 'headline'] = 99999999

    # Corruption 2: Missing mandatory attribute (NaN in 'category')
    corrupted.loc[1, 'category'] = None

    # Corruption 3: Invalid categorical value not in 42 allowable categories
    corrupted.loc[2, 'category'] = "FAKE_NEWS_UNAUTHORIZED_CATEGORY"

    # Corruption 4: Corrupted record - invalid date format
    corrupted.loc[3, 'date'] = "2026-99-99"

    # Corruption 5: Corrupted record - invalid URL link
    corrupted.loc[4, 'link'] = "ftp://corrupted-protocol-not-http"

    # Corruption 6: Schema inconsistency - unexpected extra column
    corrupted['unauthorized_malicious_column'] = "unexpected_data"

    return corrupted

def validate_schema(df: pd.DataFrame, strict: bool = True, output_report_name: str = "schema_validation_failures.csv"):
    """
    Validates the input DataFrame against the defined schema using Pandera.
    Captures all errors lazily and exports an audit report.
    """
    print(f"[DATA VALIDATION] Validating schema across {len(df)} records (strict={strict})...")
    schema = get_news_category_schema(strict=strict)

    try:
        schema.validate(df, lazy=True)
        print("[DATA VALIDATION] [SUCCESS] All records passed schema validation successfully!")
        return True, pd.DataFrame()

    except pa.errors.SchemaErrors as err:
        print("[DATA VALIDATION] [DETECTED] Schema inconsistencies and corruptions caught by Pandera!")
        
        failures = err.failure_cases
        print("\n--- Failure Cases Summary ---")
        summary_cols = [c for c in ['schema_context', 'column', 'check', 'failure_case', 'index'] if c in failures.columns]
        print(failures[summary_cols].to_string())

        report_path = os.path.join(ARTIFACTS_DIR, output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"\n[DATA VALIDATION] Detailed audit report saved to: {report_path}")
        return False, failures
