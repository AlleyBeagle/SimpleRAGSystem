# ==================================================
# 一个多Agent的智能客服系统
# ==================================================
import json
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# ==================== 0.安全解析JSON的函数 ====================
def safe_parse_json(text: str, default: dict = None) -> dict:
    """
    安全解析JSON文本

    处理：
    - Markdown代码块
    - 前后空格
    - 解析失败时返回默认值
    """
    if default is None:
        default = {}
    content = text.strip()

    # 移除Markdown代码块
    if "```json" in content:
        try:
            content = content.split("```json")[1].split("```")[0]
        except IndexError:
            pass
    elif "```" in content:
        try:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        except IndexError:
            pass

    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败：{e}")
        return default

# ==================== 0.环境变量和模型 ====================

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("未找到DEEPSEEK_API_KEY。")
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.1
)

# ==================== 0.测试数据 ====================

MOCK_ORDERS = {
    "ORD001": {
        "status": "已发货",
        "product": "智能手表 Pro",
        "price": 1299,
        "shipping": "顺丰快递",
        "tracking": "SF1234567890",
        "estimated_delivery": "2024-12-20"
    },
    "ORD002": {
        "status": "处理中",
        "product": "无线耳机 Max",
        "price": 899,
        "shipping": "待发货",
        "tracking": None,
        "estimated_delivery": "2024-12-22"
    },
    "ORD003": {
        "status": "已完成",
        "product": "便携充电宝",
        "price": 199,
        "shipping": "已签收",
        "tracking": "YT9876543210",
        "estimated_delivery": "2024-12-15"
    }
}

MOCK_PRODUCTS = {
    "智能手表 Pro": {
        "price": 1299,
        "features": ["心率监测", "GPS定位", "防水50米", "7天续航"],
        "stock": 50,
        "rating": 4.8
    },
    "无线耳机 Max": {
        "price": 899,
        "features": ["主动降噪", "40小时续航", "蓝牙5.3", "通话降噪"],
        "stock": 120,
        "rating": 4.6
    },
    "便携充电宝": {
        "price": 199,
        "features": ["20000mAh", "快充支持", "双USB输出", "LED显示"],
        "stock": 200,
        "rating": 4.5
    },
    "智能音箱": {
        "price": 499,
        "features": ["语音控制", "多房间音频", "智能家居联动", "Hi-Fi音质"],
        "stock": 80,
        "rating": 4.7
    }
}

FAQ_DATABASE = {
    "连接问题": "请尝试以下步骤：1) 重启设备 2) 检查蓝牙是否开启 3) 删除配对记录后重新配对 4) 确保设备电量充足",
    "充电问题": "建议使用原装充电器，检查充电线是否损坏。如果问题持续，可能需要更换电池或送修。",
    "软件更新": "打开设备对应的APP，进入设置-关于-检查更新，按提示操作即可完成更新。",
    "退货政策": "我们支持7天无理由退货，30天内有质量问题可换货。请保留好购买凭证和完整包装。"
}

# ==================== 1. 工具定义 ====================

@tool
def query_order(order_id: str) -> str:
    """
    根据订单号，查询订单信息

    :param order_id: 订单号，格式如ORD001
    :return: 订单详情的JSON格式的字符串
    """
    order = MOCK_ORDERS.get(order_id.upper())
    if order:
        return json.dumps(order, ensure_ascii=False, indent=2)
    return f"未找到订单 {order_id}"

@tool
def track_shipping(tracking_number: str, status: str) -> str:
    """查询物流信息

    Args:
        tracking_number: 物流单号
        status: 物流状态
    Returns:
        物流状态信息
    """
    # 模拟物流信息
    tracking_list = [
        {"SF": "顺丰快递"},
        {"YT": "圆通快递"}
    ]
    for item in tracking_list:
        for track, track_name in item.items():
            if tracking_number.startswith(track):
                return f"{track_name} {tracking_number}: {status}"
    return f"未找到物流信息 {tracking_number}"






