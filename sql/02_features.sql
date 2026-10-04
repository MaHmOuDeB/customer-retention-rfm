-- Snapshot features as of a cutoff date (only data BEFORE or ON the cutoff is used) plus the outcome label:
-- did the customer place an order in the 90 days AFTER the cutoff, and how much did they spend?  Parameters: {cutoff}
CREATE OR REPLACE TABLE features AS
WITH hist AS (
    SELECT * FROM orders WHERE order_date <= DATE '{cutoff}'
),
agg AS (
    SELECT customer_id,
           DATE_DIFF('day', MAX(order_date), DATE '{cutoff}')           AS recency_days,
           COUNT(*)                                                      AS frequency,
           SUM(order_value)                                              AS monetary,
           SUM(order_value) / COUNT(*)                                   AS avg_order_value,
           DATE_DIFF('day', MIN(order_date), DATE '{cutoff}')           AS tenure_days,
           SUM(CASE WHEN order_date > DATE '{cutoff}' - INTERVAL 90 DAY  THEN 1 ELSE 0 END)           AS orders_last_90d,
           SUM(CASE WHEN order_date > DATE '{cutoff}' - INTERVAL 90 DAY  THEN order_value ELSE 0 END) AS revenue_last_90d,
           AVG(n_products)                                               AS avg_basket_products,
           MAX(CASE WHEN country = 'United Kingdom' THEN 1 ELSE 0 END)   AS is_uk
    FROM hist GROUP BY customer_id
),
ret AS (
    SELECT customer_id, SUM(return_value) AS returned_value
    FROM returns WHERE return_date <= DATE '{cutoff}' GROUP BY customer_id
),
label AS (
    SELECT customer_id, 1 AS repurchased_90d, SUM(order_value) AS revenue_next_90d
    FROM orders
    WHERE order_date > DATE '{cutoff}' AND order_date <= DATE '{cutoff}' + INTERVAL 90 DAY
    GROUP BY customer_id
)
SELECT a.*,
       COALESCE(r.returned_value, 0) / NULLIF(a.monetary, 0) AS return_ratio,
       COALESCE(l.repurchased_90d, 0)                        AS repurchased_90d,
       COALESCE(l.revenue_next_90d, 0)                       AS revenue_next_90d
FROM agg a
LEFT JOIN ret r USING (customer_id)
LEFT JOIN label l USING (customer_id)
ORDER BY a.customer_id;   -- fixed row order: ties in the quantile scores must break the same way on every run
