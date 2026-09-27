"""
Clinical trials module (ClinicalTrials.gov). Loaded after corporate.SCHEMA_SQL.
"""

SCHEMA_SQL = r"""
-- --------------------------------------------------------------------------
-- Clinical trials, one row per trial
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clinical_trials (
    trial_id                 VARCHAR PRIMARY KEY,  -- NCT number
    corporate_id             INTEGER NOT NULL,     -- sponsor
    asset                    VARCHAR,
    indication               VARCHAR,
    phase                    VARCHAR,
    status                   VARCHAR,
    start_date               DATE,
    primary_completion_date  DATE,
    enrollment               INTEGER,
    last_updated             DATE
);
"""
