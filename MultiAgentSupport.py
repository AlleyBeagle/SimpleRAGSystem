# ==================================================
# 一个多Agent的智能客服系统
# ==================================================
from datetime import datetime
import json
import os
from typing import TypedDict, List, Dict, Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
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
    "deepseek:deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.1,
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
def track_shipping(tracking_number: str) -> str:
    """查询物流信息

    Args:
        tracking_number: 物流单号
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
                return f"{track_name} {tracking_number}: 已签收。"
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

        def quality_check(state: CustomerServiceState) -> CustomerServiceState:
            """ 质量检查 """
            print("检查生成内容的置信度...")
            result = self.qualityChecker.check(
                state["user_message"],
                state["agent_response"]
            )
            state["quality_score"] = result.get("total_score", 0) / 100

            # 质量太低需要转人工
            if result.get("needs_human_operator", False) or state["quality_score"] < 0.5:
                state["needs_human_operator"] = True
                state["human_operator_reason"] = result.get("reason", "质量检验未通过")

            print(f"质量评分：{state["quality_score"]:.2f}")
            return state

        def need_human_operator_handler(state: CustomerServiceState) -> Literal["final_human", "respond"]:
            """ 判断是否需要转人工 """
            if state["needs_human_operator"]:
                return "final_human"
            return "respond"

        def final_human_handler(state: CustomerServiceState) -> CustomerServiceState:
            """ 最终转人工处理 """
            original_response = state["agent_response"]
            if original_response:
                state["agent_response"] = f"""{original_response}
=======
系统提示：由于此问题需要更专业的处理，我们建议您联系人工客服以获得更好地服务。
"""
            else:
                state["agent_response"] = f"""
=======
系统提示：由于此问题需要更专业的处理，我们建议您联系人工客服以获得更好地服务。
"""
            return state

        def respond(state: CustomerServiceState) -> CustomerServiceState:
            """ 最终响应 """
            return state

        graph = StateGraph(CustomerServiceState)

        # 添加节点
        graph.add_node("classify_intent", classify_intent)
        graph.add_node("order_service", order_service_handler)
        graph.add_node("tech_support", tech_support_handler)
        graph.add_node("product_consult", product_consult_handler)
        graph.add_node("human_operator", human_operator_handler)
        graph.add_node("quality_check", quality_check)
        graph.add_node("final_human", final_human_handler)
        graph.add_node("respond", respond)

        graph.add_edge(START, "classify_intent")
        # 意图识别后添加分支
        graph.add_conditional_edges(
            "classify_intent",
            route_to_agent,
            {
                "order_service": "order_service",
                "tech_support": "tech_support",
                "product_consult": "product_consult",
                "human_operator": "human_operator",
            }
        )
        graph.add_edge("order_service", "quality_check")
        graph.add_edge("tech_support", "quality_check")
        graph.add_edge("product_consult", "quality_check")
        graph.add_edge("human_operator", END)

        # 质量检查后的条件路由
        graph.add_conditional_edges(
            "quality_check",
            need_human_operator_handler,
            {
                "final_human": "final_human",
                "respond": "respond"
            }
        )

        graph.add_edge("final_human", END)
        graph.add_edge("respond", END)

        return graph.compile()

    def handle_message(self, message: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """处理用户消息"""
        print("=" * 50)
        print(f"用户消息：{message}")
        print("=" * 50)

        initial_state = {
            "user_message": message,
            "chat_history": chat_history or [],
            "intent": "",
            "confidence": 0.0,
            "agent_response": "",
            "needs_human_operator": False,
            "human_operator_reason": "",
            "quality_score": 0.0,
            "metadata": {"timestamp": datetime.now().isoformat()}
        }

        result = self.graph.invoke(initial_state)

        return {
            "response": result["agent_response"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "quality_score": result["quality_score"],
            "needs_human_operator": result["needs_human_operator"],
        }


# ==================== 5. 主程序 ====================

def main():
    """ 演示多Agent客服系统 """

    system = CustomerServiceSystem()

    # 测试场景
    test_cases = [
        # 订单服务场景（关键词：订单、物流、什么时候到、快递）
        {
            "category": "订单服务",
            "messages": [
                "帮我查一下订单 ORD001 的物流状态",
                "我的订单什么时候能到？订单号是 ORD002",
                "ORD003 发货了吗？",
                "能查一下快递到哪了吗？单号 SF1234567890",
                "我要退货，订单 ORD001 怎么办理？"
            ]
        },
        # 技术支持场景（关键词：连不上、坏了、怎么用、设置、更新、充电慢）
        {
            "category": "技术支持",
            "messages": [
                "我的蓝牙耳机连接不上手机怎么办？",
                "手表充电很慢，是不是坏了？",
                "APP 连不上设备，显示配对失败",
                "怎么恢复出厂设置？",
                "固件更新到一半卡住了",
                "耳机左边没声音了"
            ]
        },
        # 产品咨询场景（关键词：怎么样、好用吗、多少钱、推荐、哪个好、功能、参数）
        {
            "category": "产品咨询",
            "messages": [
                "你们有什么智能手表推荐吗？预算1500左右",
                "无线耳机有什么功能？",
                "智能手表 Pro 和普通版有什么区别？",
                "这个充电宝支持快充吗？",
                "智能音箱音质怎么样？",
                "哪款耳机续航最长？",
                "¥499 的音箱和 ¥899 的耳机哪个更值得买？"
            ]
        },
        # 人工升级场景（关键词：投诉、人工、客服、垃圾、差评、情绪激动、经理）
        {
            "category": "人工升级",
            "messages": [
                "我要投诉！这是第三次出问题了！",
                "我想和你们经理谈谈",
                "转人工客服！",
                "你们这什么垃圾产品，我要退货！",
                "再解决不了我就去315投诉",
                "你们客服都是机器人吗？来个人说话！",
                "气死我了，这个问题折腾三天了"
            ]
        },
        # 边界模糊场景（测试分类准确性）
        {
            "category": "边界测试",
            "messages": [
                "我的订单 ORD002 的手表连不上手机",  # 订单+技术，应优先技术还是订单？
                "推荐一个能连蓝牙的充电宝",  # 产品+技术
                "我收到的耳机是坏的，怎么退货？",  # 售后+技术+订单
                "这手表防水吗？不小心掉水里了",  # 产品+技术
                "1500预算买什么？要能测心率的"  # 产品咨询
            ]
        }
    ]

    # 运行测试
    for test in test_cases:
        print("=" * 50)
        print(f"测试类别: {test['category']}")
        print("=" * 50)

        for message in test["messages"]:
            result = system.handle_message(message)

            print("\nAgent回复:")
            print(f"{result['response']}")
            print("\nAgent处理:")
            print(f"   - 意图: {result['intent']}")
            print(f"   - 置信度: {result['confidence']:.2f}")
            print(f"   - 质量评分: {result['quality_score']:.2f}")
            print(f"   - 是否转人工: {'是' if result['needs_human_operator'] else '否'}")
            print("=" * 50)

if __name__ == "__main__":
    main()

