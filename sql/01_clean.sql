-- Clean the raw invoice lines into `sales` (what customers kept) and `returns` (genuine returns).
-- A cancellation that reverses an order line for the same customer, product and quantity within 24 hours is
-- treated as an order that never happened: both sides are removed. Without this, a single mis-keyed order of
-- 74,215 units (cancelled the same day) counts as £77k of revenue and £77k of returns.
CREATE OR REPLACE TABLE raw AS SELECT * FROM read_parquet('data/online_retail_ii.parquet');

CREATE OR REPLACE TABLE sales_all AS
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

CREATE OR REPLACE TABLE returns_all AS
SELECT invoice,
       stock_code,
       CAST(customer_id AS BIGINT)          AS customer_id,
       CAST(invoice_ts AS TIMESTAMP)        AS invoice_ts,
       CAST(invoice_ts AS DATE)             AS return_date,
       -quantity                            AS quantity,
       -quantity * unit_price               AS return_value
FROM raw
WHERE customer_id IS NOT NULL AND invoice LIKE 'C%' AND quantity < 0 AND unit_price > 0 AND regexp_matches(stock_code, '^[0-9]{5}');

CREATE OR REPLACE TABLE reversed_pairs AS
SELECT DISTINCT s.invoice AS order_invoice, s.stock_code, r.invoice AS cancel_invoice
FROM returns_all r
JOIN sales_all s
  ON s.customer_id = r.customer_id AND s.stock_code = r.stock_code AND s.quantity = r.quantity
 AND s.invoice_ts <= r.invoice_ts AND s.invoice_ts >= r.invoice_ts - INTERVAL 1 DAY;

CREATE OR REPLACE TABLE sales AS
SELECT s.* FROM sales_all s
WHERE NOT EXISTS (SELECT 1 FROM reversed_pairs p WHERE p.order_invoice = s.invoice AND p.stock_code = s.stock_code);

CREATE OR REPLACE TABLE returns AS
SELECT r.customer_id, r.return_date, r.return_value FROM returns_all r
WHERE NOT EXISTS (SELECT 1 FROM reversed_pairs p WHERE p.cancel_invoice = r.invoice AND p.stock_code = r.stock_code);

-- One row per order
CREATE OR REPLACE TABLE orders AS
SELECT customer_id, invoice, MIN(invoice_date) AS order_date, SUM(revenue) AS order_value,
       COUNT(DISTINCT stock_code) AS n_products, ANY_VALUE(country) AS country
FROM sales GROUP BY customer_id, invoice;
