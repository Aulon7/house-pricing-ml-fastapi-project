-- Run this in a Databricks SQL editor before enabling API prediction logging.
-- If you use a different catalog/schema/table, set the same fully qualified
-- name in DATABRICKS_PREDICTIONS_TABLE inside your local .env file.
CREATE TABLE IF NOT EXISTS workspace.default.iowa_predictions (
    input_features STRING NOT NULL,
    prediction DOUBLE NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    model_version STRING NOT NULL
)
USING DELTA;

-- After calling POST /predict, use this to verify that a row was written:
-- SELECT *
-- FROM workspace.default.iowa_predictions
-- ORDER BY prediction_timestamp DESC
-- LIMIT 10;
