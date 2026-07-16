# Data Collection Notes

这个目录用于保存电商智能客服 Agent 的公开数据和清洗后数据。

## 当前步骤

当前先把 Agent 需要的核心数据表落地：

- `raw/makeup_api_products.json`: Makeup API 原始 JSON。
- `raw/openbeautyfacts_products.json`: Open Beauty Facts 原始 JSON，清洗时只保留带成分表的护肤记录。
- `raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv`: Bitext retail/eCommerce 客服意图训练集原始 CSV。
- `raw/*policies*.html`: 公开品牌政策页原始 HTML。
- `processed/product_knowledge.csv`: 统一字段后的商品知识表。
- `processed/faq_knowledge.csv`: 从公开客服意图训练集整理出的 FAQ/标准问答表。
- `processed/policy_knowledge.csv`: 从公开品牌售后政策页整理出的售后政策表。
- `processed/shipping_rules.csv`: 从公开品牌物流政策页整理出的物流规则表。
- `processed/order_mock_data.csv`: 基于公开商品数据生成的模拟订单表。
- `processed/source_manifest.json`: 数据来源、采集时间和行数说明。
- `processed/service_data_manifest.json`: FAQ、政策、物流和 mock 订单的数据来源说明。

## 合规边界

- 只采集公开商品/标签数据。
- 不采集真实用户订单、手机号、地址、昵称、评价账号等个人信息。
- Open Beauty Facts API 请求带自定义 `User-Agent`，并控制请求频率。
- 真实订单数据后续只使用 mock data。
- FAQ 使用公开合成/标注数据集，不采集真实私聊记录。
- 对返回 403、Access Denied 或 robots 不允许的页面不做采集。

## 使用方式

运行 `es_store/es_ingest.py --sources all` 时，系统会读取本目录的清洗结果，统一转换为知识文档并增量写入 Elasticsearch。订单数据不会进入向量库，只由业务工具按订单号查询。

## 字段说明

`processed/product_knowledge.csv` 的核心字段：

- `product_id`: 带来源前缀的唯一 ID。
- `source`: 数据来源。
- `source_url`: 来源入口。
- `product_url`: 商品详情页或来源页。
- `name`, `brand`, `category`, `sub_category`: 商品基础信息。
- `price`, `currency`, `specification`, `colors`: 价格、规格、色号等。
- `selling_points`, `detail`, `efficacy`: 用于商品问答和推荐的文本字段。
- `applicable_people`, `not_suitable_people`, `cautions`: Agent 回答时的安全边界。
- `ingredients`: 护肤品成分字段，主要来自 Open Beauty Facts。

其他表：

- `faq_knowledge.csv`: `question`, `answer`, `intent`, `category`, `priority`, `need_human`。
- `policy_knowledge.csv`: `policy_type`, `return_rule`, `exchange_rule`, `refund_rule`, `freight_rule`, `special_limits`。
- `shipping_rules.csv`: `dispatch_time`, `free_shipping_threshold`, `carriers`, `estimated_delivery_time`, `abnormal_shipping_handling`。
- `order_mock_data.csv`: `order_status`, `payment_status`, `fulfillment_status`, `tracking_number`, `is_cancelable`, `is_refundable`。

原始公开数据保留在 `raw/`，用于来源追踪；线上 Agent 只读取 `processed/` 中的清洗结果。
