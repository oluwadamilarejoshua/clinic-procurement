SELECT 
    -- Purchase Order Info
    po.id AS order_id,
    po.name AS purchase_order_name,
    po.date_order,
    po.state,

    -- Vendor
    rp.name AS vendor_name,

    -- Branch
    eb.id AS branch_id,
    eb.name AS branch_name,

    -- Product Info
    get_json_object(pt.name, '$.en_US') AS product_name,
    pol.name AS line_description,

    -- Category
    pc.name AS category_name,

    -- Metrics
    pol.product_qty,
    pol.price_unit,
    pol.price_subtotal

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
