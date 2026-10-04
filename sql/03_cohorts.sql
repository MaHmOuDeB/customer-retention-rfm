-- Monthly acquisition cohorts and the share of each cohort that orders again in month N
CREATE OR REPLACE TABLE cohort_retention AS
WITH first_order AS (
    SELECT customer_id, DATE_TRUNC('month', MIN(order_date)) AS cohort_month FROM orders GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT o.customer_id, f.cohort_month,
           DATE_DIFF('month', f.cohort_month, DATE_TRUNC('month', o.order_date)) AS month_n
    FROM orders o JOIN first_order f USING (customer_id)
),
size AS (SELECT cohort_month, COUNT(*) AS cohort_size FROM first_order GROUP BY cohort_month)
SELECT a.cohort_month, a.month_n, s.cohort_size, COUNT(*) AS active_customers,
       COUNT(*) * 1.0 / s.cohort_size AS retention
FROM activity a JOIN size s USING (cohort_month)
GROUP BY a.cohort_month, a.month_n, s.cohort_size
ORDER BY 1, 2;
