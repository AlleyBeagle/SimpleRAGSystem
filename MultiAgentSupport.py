# ==================================================
# 一个多Agent的智能客服系统
# ==================================================
import json
import os
from typing import TypedDict, List, Dict, Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

import PROMPT_TEMPLATE


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
os.environ['LANGGRAPH_ALLOWED_OBJECTS'] = 'messages'    # 这个是为了解决LangChainPendingDeprecationWarning

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

@tool
def search_product(keyword: str) -> str:
    """
    根据特征/关键词搜索产品信息
    :param keyword: 产品关键词
    :return: 匹配到产品的信息
    """
    results = []
    for name, info in MOCK_PRODUCTS.items():
        if keyword.lower() in name.lower():
            results.append({
                "name": name,
                "price": f"¥{info['price']}",
                "features": info["features"],
                "rating": f"{info["rating"]}分",
            })

        if results:
            return json.dumps(results, ensure_ascii=False, indent=2)
        return f"未找到包含 '{keyword}' 的商品。"

@tool
def get_product_recommendations(budget: int, category: str = "全部") -> str:
    """
    根据预算推荐产品
    :param budget: 预算金额
    :param category: 产品类别（可选）
    :return: 推荐产品列表
    """
    recommendations = []
    for name, info in MOCK_PRODUCTS.items():
        if info["price"] <= budget:         # 这里暂且是小于预算金额的都要
            recommendations.append({
                "name": name,
                "price": f"¥{info['price']}",
                "rating": info["rating"],
            })

        # 按评分排序
        recommendations.sort(key=lambda x: x["rating"], reverse=True)

        if recommendations:
            return json.dumps(recommendations, ensure_ascii=False, indent=2)
        return f"在预算 ¥{budget} 内暂无推荐产品。"

@tool
def search_faq(problem_type: str) -> str:
    """
    所有常见问题解答
    :param problem_type: 问题类型关键词
    :return: 相关FAQ答案
    """
    for key, answer in FAQ_DATABASE.items():
        if problem_type in key or key in problem_type:
            return f"【{key}】\n{answer}"
        return "未找到相关FAQ，建议联系人工客服获取帮助。"


# ==================== 2. 状态定义 ====================

class CustomerServiceState(TypedDict):
    """ 客服系统状态 """
    user_message: str                   # 用户消息
    chat_history: List[Dict[str, str]]  # 对话历史
    intent: str                         # 识别的意图
    confidence: float                   # 意图置信度
    agent_response: str                 # Agent回复
    needs_human_operator: bool          # 是否需要转人工
    human_operator_reason: str          # 转人工原因
    quality_score: float                # 质量评分
    metadata: Dict[str, Any]            # 元数据


# ==================== 3. 多个Agent定义 ====================

# 3.1 意图分类器
class IntentClassifier:
    """ 意图分类器 """
    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE.INTENT_PROMPT),
            ("human", "{message}")
        ])

    def classify(self, message: str) -> Dict[str, Any]:
        """ 将用户意图分类，传给不同的Agent """
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({"message": message})

        # 使用安全的JSON解析
        default_result = {"intent": "human_operator", "confidence": 0.5, "reason": "解析失败"}
        parsed = safe_parse_json(result, default_result)

        if "intent" not in parsed:
            return default_result
        return parsed


# 3.2 订单处理Agent
class OrderServiceAgent:
    """ 订单处理Agent """

    def __init__(self):
        self.llm = model
        self.tools = [query_order, track_shipping]
        self.system_prompt = PROMPT_TEMPLATE.ORDER_SERVICE_PROMPT
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

        def handle(self, message: str, chat_history: List = None) -> str:
            """ 处理订单服务请求 """
            messages = [{"role": "user", "content": message}]
            result = self.agent.invoke({"messages": messages})

            # 提取最终回复
            if result["messages"]:
                return result["messages"][-1].content
            return "抱歉，订单查询服务暂不可用，请稍候再试。"


# 3.3 技术支持Agent
class TechSupportAgent:
    """ 技术支持Agent """

    def __init__(self):
        self.llm = model
        self.tools = [search_faq]
        self.system_prompt = PROMPT_TEMPLATE.TECH_SUPPORT_PROMPT
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        """ 处理技术支持请求 """
        messages = [{"role": "user", "content": message}]
        result = self.agent.invoke({"messages": messages})

        # 提取最终回复
        if result["messages"]:
            return result["messages"][-1].content
        return "抱歉，我无法处理您的问题，建议联系人工客服。"


# 3.4 产品咨询Agent
class ProductConsultAgent:
    """ 产品咨询Agent """

    def __init__(self):
        self.llm = model
        self.tools = [search_product, get_product_recommendations]
        self.system_prompt = PROMPT_TEMPLATE.PRODUCT_CONSULT_PROMPT
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        """ 处理技术支持请求 """
        messages = [{"role": "user", "content": message}]
        result = self.agent.invoke({"messages": messages})

        # 提取最终回复
        if result["messages"]:
            return result["messages"][-1].content
        return "抱歉，产品咨询服务暂不可用，请稍候再试。"


# 3.5 生成内容质量检查器
class QualityChecker:
    """ 质量检查器（是否需要转人工） """

    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE.QUALITER_CHECKER_PROMPT),
            ("human", """用户问题：{user_message}
客服回复：{agent_response}
评估结果：
            """)
        ])

    def check(self, user_message: str, agent_response: str) -> Dict[str, Any]:
        """检查回复质量"""
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "user_message": user_message,
            "agent_response": agent_response
        })

        # 使用安全的 JSON 解析
        default_result = {"total_score": 60, "needs_human_operator": False, "reason": "评估完成"}
        return safe_parse_json(result, default_result)


# ==================== 4. 客服系统主类 ====================

class CustomerServiceSystem:
    """ 多Agent客服系统 """

    def __init__(self):
        # 初始化组件
        self.intentClassifier = IntentClassifier()
        self.orderServiceAgent = OrderServiceAgent()
        self.techSupportAgent = TechSupportAgent()
        self.productConsultAgent = ProductConsultAgent()
        self.qualityChecker = QualityChecker()

        # 构建图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """ 构建 Langgraph 工作流 """

        def classify_intent(state: CustomerServiceState) -> CustomerServiceState:
            """ 分类用户意图 """
            result = self.intentClassifier.classify(state["user_message"])

            state["intent"] = result.get("intent", "human_operator")       # 不知道就转人工
            state["confidence"] = result.get("confidence", 0.5)

            print(f"意图：{state["intent"]} （置信度：{state['confidence']:.2f}）")
            return state

        def route_to_agent(state: CustomerServiceState) -> Literal["tech_support", "order_service", "product_consult", "human_operator"]:
            """ 路由到对应的Agent """
            intent = state["intent"]
            confidence = state["confidence"]

            # 低置信度直接转人工，其余转到对应意图的Agent
            if confidence < 0.5:
                return "human_operator"
            if intent == "order_service":
                return "order_service"
            elif intent == "tech_support":
                return "tech_support"
            elif intent == "product_consult":
                return "product_consult"
            else:
                return "human_operator"

        def order_service_handler(state: CustomerServiceState) -> CustomerServiceState:
            """ 订单服务处理 """
            print("订单服务Agent运行中...")
            response = self.orderServiceAgent.handle(state["user_message"])
            state["agent_response"] = response
            return state

        def tech_support_handler(state: CustomerServiceState) -> CustomerServiceState:
            """ 技术支持处理 """
            print("技术支持Agent运行中...")
            response = self.techSupportAgent.handle(state["user_message"])
            state["agent_response"] = response
            return state

        def product_consult_handler(state: CustomerServiceState) -> CustomerServiceState:
            """ 产品咨询处理 """
            print("产品咨询Agent运行中...")
            response = self.productConsultAgent.handle(state["user_message"])
            state["agent_response"] = response
            return state

        def human_operator_handler(state: CustomerServiceState) -> CustomerServiceState:
            """升级处理"""
            print("转人工...")
            state["needs_human_operator"] = True
            state["human_operator_reason"] = "意图识别置信度低或用户要求人工服务"
            state["agent_response"] = """非常抱歉，您的问题需要人工客服来处理。
我已经为您转接人工客服，请稍候...

在等待期间，您也可以：
1. 拨打客服热线：400-xxx-xxxx
2. 发送邮件至：support@example.com
3. 在XX板块下留言确认问题
4. 工作日 9:00-18:00 在线客服响应更快

感谢您的耐心等待！"""
            return state


