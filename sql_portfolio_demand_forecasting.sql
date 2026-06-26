-- ============================================================
-- SQL PORTFOLIO: E-Commerce Demand Forecasting
-- Dataset: Shopee Sales Data (Dec 2023 – Nov 2025)
-- Total Records: ~20,848 orders
-- Author: [Nama Lo]
-- ============================================================


-- ============================================================
-- SECTION 1: DATA PREPARATION & CLEANING
-- Tujuan: Normalisasi data mentah sebelum masuk ke modeling
-- ============================================================

-- [Q1] Standardisasi status pesanan menjadi kategori bersih
-- Raw data punya status panjang seperti "Pesanan diterima, namun Pembeli masih..."
-- Di-map jadi 4 kategori bersih untuk analisis

CREATE VIEW clean_orders AS
SELECT
    order_id,
    total_qty,
    total_weight_gr,
    total_returned_qty,
    "Total Diskon"            AS total_discount,
    product_categories,
    num_product_categories,
    "Opsi Pengiriman"         AS shipping_option,
    "Metode Pembayaran"       AS payment_method,
    "Kota/Kabupaten"          AS city,
    "Provinsi"                AS province,
    "Ongkos Kirim Dibayar oleh Pembeli"     AS shipping_paid_by_buyer,
    "Estimasi Potongan Biaya Pengiriman"    AS shipping_discount,
    "Total Pembayaran"        AS total_revenue,
    "Perkiraan Ongkos Kirim"  AS estimated_shipping_cost,
    CAST("Waktu Pesanan Dibuat" AS TIMESTAMP) AS order_time,
    CASE
        WHEN "Status Pesanan" = 'Selesai'                        THEN 'Completed'
        WHEN "Status Pesanan" = 'Batal'                          THEN 'Cancelled'
        WHEN "Status Pesanan" LIKE 'Pesanan diterima%'           THEN 'Delivered'
        WHEN "Status Pesanan" IN ('Sedang Dikirim','Telah Dikirim') THEN 'In Transit'
        ELSE 'Other'
    END AS order_status,
    source_file
FROM raw_orders;


-- ============================================================
-- SECTION 2: FEATURE ENGINEERING (untuk input model ML)
-- Tujuan: Ekstrak fitur temporal dari timestamp
-- Ini yang nantinya jadi kolom X di Prophet/LSTM
-- ============================================================

-- [Q2] Ekstrak fitur waktu lengkap per order
-- Output: dataset siap pakai untuk Prophet (kolom ds & y)
-- dan fitur tambahan untuk LSTM

SELECT
    order_id,
    order_time,
    total_qty,
    total_revenue,
    order_status,
    product_categories,
    province,

    -- Fitur temporal
    DATE(order_time)                            AS order_date,
    EXTRACT(YEAR   FROM order_time)             AS year,
    EXTRACT(MONTH  FROM order_time)             AS month,
    EXTRACT(DAY    FROM order_time)             AS day_of_month,
    EXTRACT(DOW    FROM order_time)             AS day_of_week,    -- 0=Sun, 6=Sat
    EXTRACT(HOUR   FROM order_time)             AS hour_of_day,
    EXTRACT(WEEK   FROM order_time)             AS week_of_year,

    -- Fitur derived
    CASE WHEN EXTRACT(DOW FROM order_time) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN EXTRACT(HOUR FROM order_time) BETWEEN 7  AND 11 THEN 'morning'
         WHEN EXTRACT(HOUR FROM order_time) BETWEEN 12 AND 17 THEN 'afternoon'
         WHEN EXTRACT(HOUR FROM order_time) BETWEEN 18 AND 22 THEN 'evening'
         ELSE 'night' END AS time_of_day,

    -- Flash sale / payday proxy (tanggal 1, 10, 15, 25 = tanggal umum promo Shopee)
    CASE WHEN EXTRACT(DAY FROM order_time) IN (1, 10, 15, 25) THEN 1 ELSE 0 END AS is_promo_date,

    -- Double-date sale proxy (1/1, 2/2, 3/3, ... 11/11, 12/12)
    CASE WHEN EXTRACT(MONTH FROM order_time) = EXTRACT(DAY FROM order_time) THEN 1 ELSE 0 END AS is_double_date

FROM clean_orders
WHERE order_status = 'Completed'
ORDER BY order_time;


-- ============================================================
-- SECTION 3: AGGREGASI HARIAN (Input utama model forecast)
-- Tujuan: Buat time series harian — ini langsung masuk Prophet
-- Format: ds (date), y (demand)
-- ============================================================

-- [Q3] Daily demand aggregation — siap masuk Prophet
SELECT
    DATE(order_time)        AS ds,
    COUNT(*)                AS total_orders,        -- y: volume order
    SUM(total_qty)          AS total_units_sold,    -- alternatif y: unit terjual
    SUM(total_revenue)      AS daily_revenue,
    AVG(total_revenue)      AS avg_order_value,
    SUM(total_discount)     AS total_discount_given,
    COUNT(DISTINCT product_categories) AS unique_categories

FROM clean_orders
WHERE order_status = 'Completed'
GROUP BY DATE(order_time)
ORDER BY ds;


-- [Q4] Daily demand per province — untuk forecast per zona
-- Berguna jika ingin forecast berbeda per wilayah (Jawa Barat vs DKI, dll)
SELECT
    DATE(order_time)    AS ds,
    province,
    COUNT(*)            AS total_orders,
    SUM(total_qty)      AS total_units,
    SUM(total_revenue)  AS revenue

FROM clean_orders
WHERE order_status = 'Completed'
GROUP BY DATE(order_time), province
ORDER BY ds, total_orders DESC;


-- ============================================================
-- SECTION 4: PATTERN DISCOVERY (EDA via SQL)
-- Tujuan: Temukan seasonality sebelum modeling
-- Output: insight yang di-plot di EDA notebook
-- ============================================================

-- [Q5] Demand heatmap: hour x day_of_week
-- Menampilkan jam & hari tersibuk — dasar untuk peak hour analysis
SELECT
    EXTRACT(DOW  FROM order_time) AS day_of_week,   -- 0=Sun
    EXTRACT(HOUR FROM order_time) AS hour_of_day,
    COUNT(*)                       AS order_count,
    ROUND(AVG(total_revenue), 0)   AS avg_revenue

FROM clean_orders
WHERE order_status = 'Completed'
GROUP BY 1, 2
ORDER BY 1, 2;


-- [Q6] Monthly trend — identifikasi seasonality bulanan
SELECT
    TO_CHAR(order_time, 'YYYY-MM')              AS year_month,
    EXTRACT(YEAR  FROM order_time)              AS year,
    EXTRACT(MONTH FROM order_time)              AS month,
    COUNT(*)                                    AS total_orders,
    SUM(total_revenue)                          AS total_revenue,
    SUM(total_qty)                              AS total_units,
    ROUND(AVG(total_revenue), 0)                AS avg_order_value,
    COUNT(*) FILTER (WHERE order_status = 'Cancelled')  AS cancelled_orders,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE order_status = 'Cancelled') / COUNT(*), 2
    )                                           AS cancellation_rate_pct

FROM clean_orders
GROUP BY 1, 2, 3
ORDER BY 1;


-- [Q7] Demand per hari dalam seminggu — weekly seasonality
SELECT
    EXTRACT(DOW FROM order_time)    AS day_num,
    TO_CHAR(order_time, 'Day')      AS day_name,
    COUNT(*)                        AS total_orders,
    ROUND(AVG(COUNT(*)) OVER(), 0)  AS avg_daily_orders,   -- window function
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_weekly_demand

FROM clean_orders
WHERE order_status = 'Completed'
GROUP BY 1, 2
ORDER BY 1;


-- ============================================================
-- SECTION 5: PRODUCT & CATEGORY ANALYSIS
-- Tujuan: Forecast demand per kategori produk
-- ============================================================

-- [Q8] Top 10 kategori by revenue + volume + return rate
SELECT
    product_categories,
    COUNT(*)                                                    AS total_orders,
    SUM(total_qty)                                              AS total_units_sold,
    SUM(total_revenue)                                          AS total_revenue,
    ROUND(AVG(total_revenue), 0)                                AS avg_order_value,
    SUM(total_returned_qty)                                     AS total_returned,
    ROUND(100.0 * SUM(total_returned_qty) / NULLIF(SUM(total_qty), 0), 2) AS return_rate_pct,
    COUNT(*) FILTER (WHERE order_status = 'Cancelled')          AS cancelled,
    ROUND(100.0 * COUNT(*) FILTER (WHERE order_status = 'Cancelled') / COUNT(*), 2) AS cancel_rate_pct

FROM clean_orders
GROUP BY product_categories
ORDER BY total_revenue DESC
LIMIT 10;


-- [Q9] Monthly demand per top category — untuk multi-series forecast
-- Bisa dipakai bikin satu model forecast per kategori
SELECT
    TO_CHAR(order_time, 'YYYY-MM')  AS year_month,
    product_categories,
    COUNT(*)                        AS total_orders,
    SUM(total_qty)                  AS total_units,
    SUM(total_revenue)              AS revenue

FROM clean_orders
WHERE
    order_status = 'Completed'
    AND product_categories IN (
        -- Ambil top 5 kategori by volume
        SELECT product_categories
        FROM clean_orders
        WHERE order_status = 'Completed'
        GROUP BY product_categories
        ORDER BY COUNT(*) DESC
        LIMIT 5
    )
GROUP BY 1, 2
ORDER BY 1, 3 DESC;


-- ============================================================
-- SECTION 6: SHIPPING & PAYMENT ANALYSIS
-- Tujuan: Understand fulfillment behavior — relevan untuk
-- menghitung cost impact dari demand forecasting
-- ============================================================

-- [Q10] Shipping option preference per bulan
-- Berguna untuk model yang mau predict shipping load per metode
SELECT
    TO_CHAR(order_time, 'YYYY-MM')      AS year_month,
    shipping_option,
    COUNT(*)                            AS order_count,
    SUM(estimated_shipping_cost)        AS total_shipping_cost,
    ROUND(AVG(estimated_shipping_cost), 0) AS avg_shipping_cost,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY TO_CHAR(order_time, 'YYYY-MM')), 2) AS pct_of_month

FROM clean_orders
WHERE order_status = 'Completed'
GROUP BY 1, 2
ORDER BY 1, 3 DESC;


-- [Q11] Payment method breakdown — COD vs digital payment trend
-- Insight: COD tinggi = demand forecasting lebih volatile (cancel rate lebih tinggi)
SELECT
    payment_method,
    COUNT(*)                            AS total_orders,
    COUNT(*) FILTER (WHERE order_status = 'Cancelled') AS cancelled_orders,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE order_status = 'Cancelled') / COUNT(*), 2
    )                                   AS cancellation_rate_pct,
    SUM(total_revenue)                  AS total_revenue,
    ROUND(AVG(total_revenue), 0)        AS avg_order_value

FROM clean_orders
GROUP BY payment_method
ORDER BY total_orders DESC;


-- ============================================================
-- SECTION 7: ANOMALY DETECTION (SQL-based)
-- Tujuan: Temukan tanggal anomali sebelum masuk model
-- Anomali bisa jadi outlier yang perlu di-handle Prophet
-- ============================================================

-- [Q12] Deteksi tanggal dengan demand spike / drop ekstrem
-- Logic: hari dengan order > mean + 2*stddev dianggap anomali
WITH daily_demand AS (
    SELECT
        DATE(order_time)    AS order_date,
        COUNT(*)            AS daily_orders

    FROM clean_orders
    WHERE order_status = 'Completed'
    GROUP BY DATE(order_time)
),
stats AS (
    SELECT
        AVG(daily_orders)       AS mean_orders,
        STDDEV(daily_orders)    AS stddev_orders
    FROM daily_demand
)
SELECT
    d.order_date,
    d.daily_orders,
    ROUND(s.mean_orders, 1)                         AS mean_orders,
    ROUND(s.stddev_orders, 1)                       AS stddev_orders,
    ROUND((d.daily_orders - s.mean_orders) / NULLIF(s.stddev_orders, 0), 2) AS z_score,
    CASE
        WHEN d.daily_orders > s.mean_orders + 2 * s.stddev_orders THEN 'SPIKE'
        WHEN d.daily_orders < s.mean_orders - 2 * s.stddev_orders THEN 'DROP'
        ELSE 'Normal'
    END AS anomaly_flag

FROM daily_demand d, stats s
ORDER BY ABS((d.daily_orders - s.mean_orders) / NULLIF(s.stddev_orders, 0)) DESC;


-- ============================================================
-- SECTION 8: GEOGRAPHIC DEMAND ANALYSIS
-- Tujuan: Segmentasi demand per provinsi / kota
-- Berguna untuk operasional: prioritas wilayah pengiriman
-- ============================================================

-- [Q13] Revenue & volume per provinsi — untuk dashboard peta
SELECT
    province,
    COUNT(*)                        AS total_orders,
    SUM(total_qty)                  AS total_units,
    SUM(total_revenue)              AS total_revenue,
    ROUND(AVG(total_revenue), 0)    AS avg_order_value,
    SUM(estimated_shipping_cost)    AS total_shipping_cost,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total_orders

FROM clean_orders
WHERE order_status = 'Completed'
GROUP BY province
ORDER BY total_orders DESC;


-- [Q14] Growth rate provinsi — YoY comparison
-- Melihat provinsi mana yang tumbuh/turun demand-nya
WITH monthly_province AS (
    SELECT
        province,
        EXTRACT(YEAR FROM order_time)   AS year,
        EXTRACT(MONTH FROM order_time)  AS month,
        COUNT(*)                        AS orders

    FROM clean_orders
    WHERE order_status = 'Completed'
    GROUP BY 1, 2, 3
)
SELECT
    curr.province,
    curr.year      AS current_year,
    curr.month,
    curr.orders    AS current_orders,
    prev.orders    AS prev_year_orders,
    ROUND(100.0 * (curr.orders - prev.orders) / NULLIF(prev.orders, 0), 2) AS yoy_growth_pct

FROM monthly_province curr
LEFT JOIN monthly_province prev
    ON  curr.province = prev.province
    AND curr.month    = prev.month
    AND curr.year     = prev.year + 1

WHERE curr.year = 2025
ORDER BY curr.province, curr.month;


-- ============================================================
-- SECTION 9: BUSINESS IMPACT CALCULATION
-- Tujuan: Kuantifikasi dampak bisnis dari model forecast
-- Ini yang masuk ke CV framing: "12% reduction in idle cost"
-- ============================================================

-- [Q15] Hitung estimasi kerugian dari order yang batal
-- Cancelled orders = demand yang tidak terfulfill = shipping slot terbuang
WITH cancellation_summary AS (
    SELECT
        TO_CHAR(order_time, 'YYYY-MM')                  AS year_month,
        COUNT(*)                                         AS total_orders,
        COUNT(*) FILTER (WHERE order_status = 'Cancelled') AS cancelled_orders,
        SUM(estimated_shipping_cost)
            FILTER (WHERE order_status = 'Cancelled')   AS lost_shipping_revenue,
        SUM(total_revenue)
            FILTER (WHERE order_status = 'Completed')   AS realized_revenue

    FROM clean_orders
    GROUP BY 1
)
SELECT
    year_month,
    total_orders,
    cancelled_orders,
    ROUND(100.0 * cancelled_orders / total_orders, 2)           AS cancel_rate_pct,
    lost_shipping_revenue,
    realized_revenue,
    -- Simulasi: jika cancel rate turun 20% berkat forecast lebih akurat
    ROUND(lost_shipping_revenue * 0.20, 0)                      AS potential_saving_20pct_reduction

FROM cancellation_summary
ORDER BY year_month;


-- [Q16] Demand forecast readiness check — validasi kualitas data time series
-- Cek gap tanggal sebelum masuk Prophet (Prophet butuh data kontinu)
WITH date_series AS (
    SELECT generate_series(
        (SELECT MIN(DATE(order_time)) FROM clean_orders),
        (SELECT MAX(DATE(order_time)) FROM clean_orders),
        INTERVAL '1 day'
    )::DATE AS calendar_date
),
actual_dates AS (
    SELECT DISTINCT DATE(order_time) AS order_date
    FROM clean_orders
    WHERE order_status = 'Completed'
)
SELECT
    d.calendar_date,
    CASE WHEN a.order_date IS NULL THEN 'MISSING' ELSE 'OK' END AS data_status

FROM date_series d
LEFT JOIN actual_dates a ON d.calendar_date = a.order_date
WHERE a.order_date IS NULL   -- hanya tampilkan tanggal yang missing
ORDER BY d.calendar_date;
