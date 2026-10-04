-- Clean sales lines. Cancellations (invoice starts with 'C') are kept separately as returns.
CREATE OR REPLACE TABLE raw AS SELECT * FROM read_parquet('data/online_retail_ii.parquet');

CREATE OR REPLACE TABLE sales AS
SELECT invoice,
       stock_code,
       CAST(customer_id AS BIGINT)          AS customer_id,
       country,
       CAST(invoice_ts AS TIMESTAMP)        AS invoice_ts,
       CAST(invoice_ts AS DATE)             AS invoice_date,
       quantity,
       unit_price,
       quantity * unit_price                AS revenue
FROM raw
WHERE customer_id IS NOT NULL
  AND invoice NOT LIKE 'C%'
  AND quantity > 0 AND unit_price > 0
  AND regexp_matches(stock_code, '^[0-9]{5}');        -- real products only (drops POST, DOT, M, BANK CHARGES ...)

CREATE OR REPLACE TABLE returns AS
SELECT CAST(customer_id AS BIGINT) AS customer_id,
       CAST(invoice_ts AS DATE)    AS return_date,
       -quantity * unit_price      AS return_value
FROM raw
WHERE customer_id IS NOT NULL AND invoice LIKE 'C%' AND regexp_matches(stock_code, '^[0-9]{5}');

-- One row per order
CREATE OR REPLACE TABLE orders AS
SELECT customer_id, invoice, MIN(invoice_date) AS order_date, SUM(revenue) AS order_value,
       COUNT(DISTINCT stock_code) AS n_products, ANY_VALUE(country) AS country
FROM sales GROUP BY customer_id, invoice;
