SELECT
    -- Purchase Order Info
    po.id AS order_id,
    po.name AS purchase_order_name,
    po.date_order,
    po.state,

    -- Branch
    eb.name AS branch_name,

    -- Product
    pt.id AS product_id,
      get_json_object(pt.name, '$.en_US') AS product_name,
    pol.name AS line_description,

    -- Category
    pc.name AS category_name,

    -- NEW: Consumables grouping
    CASE
        WHEN pc.name IN (
            'Consumables',
            'Medical Consumables',
            'Dental Consumables',
            'Laboratory Consumables'
        )
        THEN 'Consumables'
        ELSE 'Others'
    END AS category_group,

    -- Metrics
    pol.product_qty,
    pol.price_unit,
    pol.price_subtotal,

    -- Time fields (useful for BI)
    DATE_TRUNC('month', po.date_order) AS month,
    CONCAT(
        'Q',
        EXTRACT(QUARTER FROM po.date_order),
        ', ',
        EXTRACT(YEAR FROM po.date_order)
    ) AS quarter,
    EXTRACT(YEAR FROM po.date_order) AS year

FROM purchase_order po

LEFT JOIN eha_branch eb
    ON eb.id = po.branch_id

JOIN purchase_order_line pol
    ON pol.order_id = po.id

JOIN product_product pp
    ON pp.id = pol.product_id

JOIN product_template pt
    ON pt.id = pp.product_tmpl_id

JOIN product_category pc
    ON pc.id = pt.categ_id

LEFT JOIN res_partner rp
    ON rp.id = po.partner_id

WHERE po.state IN ('purchase','done');