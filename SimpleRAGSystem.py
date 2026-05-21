# ==================================================
# 一个简单的RAG项目，基于Langchain和Langgraph搭建。
# ==================================================
import os
from dataclasses import dataclass
from typing import List, TypedDict, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph

import PROMPT_TEMPLATE

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 这里的DASHSCOPE_API_KEY对应qwen模型，用于处理多模态任务
if not DASHSCOPE_API_KEY:
    raise ValueError("未找到DASHSCOPE_API_KEY。")
if not DEEPSEEK_API_KEY:
    raise ValueError("未找到DEEPSEEK_API_KEY。")
model = init_chat_model(
    "deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.1
)

# ==================== 0. 试用Embeddings，没用 ====================
class SimpleEmbeddings(Embeddings):
    """
    简单的Embeddings实现
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """ 嵌入文档列表 """
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """ 嵌入查询 """
        return self._embed_text(text)

    def _embed_text(self, text):
        """ 简单的文本嵌入（基于字符频率）"""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # 扩展到目标维度
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            # 归一化到[-1, 1]
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            vector.append(value)
        return vector

def get_embeddings():
    """ 选择阿里的text-embedding-v1模型 """
    if DASHSCOPE_API_KEY and DASHSCOPE_API_KEY != "your_dashscope_api_key_here":
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            print("使用DashScopeEmbeddings：")
            return DashScopeEmbeddings(
                model="text-embedding-v1",
                dashscope_api_key=DASHSCOPE_API_KEY,
            )
        except Exception as e:
            print(f"Error: {e}")

    return SimpleEmbeddings()

# ==================== 1.1 RAG配置 ====================
@dataclass
class RAGConfig:
    """ RAG系统配置 """
    # 模型配置（使用全局 model）
    temperature: float = 0.1

    # 分块配置
    chunk_size: int = 500
    chunk_overlap: int = 50

    # 检索配置
    top_k: int = 3
    search_type: str = "similarity"

    # 生成配置
    max_tokens: int = 1000

# ==================== 1.2 状态定义 ====================

class RAGState(TypedDict):
    """ RAG流程状态 """
    query: str                          # 用户query
    chat_history: List[Dict[str, str]]  # 对话历史
    documents: List[Document]           # 检索到的文档
    context: str                        # 格式化的上下文
    answer: str                         # 生成的回答
    sources: List[Dict[str, Any]]       # 来源信息
    confidence: float                   # 置信度评分

# ==================== 2. 文档处理模块 ====================

class DocumentProcessor:
    """ 文档处理器：加载、切块、向量化 """
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        self.embeddings = get_embeddings()
        self.vector_store = None

    def load_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """ 从文本创建文档 """
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {"source": f"doc_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """ 文档分块 """
        return self.text_splitter.split_documents(documents)

    # def create_vector_store(self, documents: List[Document]) -> InMemoryVectorStore:
    #     """ 创建向量存储（内存） """
    #     self.vector_store = InMemoryVectorStore.from_documents(
    #         documents=documents,
    #         embedding=self.embeddings,
    #     )
    #     return self.vector_store

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """ 创建可持久化的FAISS存储 """
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        return self.vector_store

    def process(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> FAISS:
        """ 完整流程：加载、分块、向量化 """
        print("process: 加载文档")
        documents = self.load_documents(texts, metadatas)
        print(f"加载了 {len(documents)} 个文档。")

        print("process: 文档分块")
        chunks = self.split_documents(documents)
        print(f"生成了 {len(chunks)} 个文本块")

        print("process: 创建向量存储")
        vector_store = self.create_vector_store(chunks)
        print("向量存储创建完成。")

        return vector_store

    # ==================== 以下是FAISS的持久化方法 ====================

    def save_vector_store(self, path: str):
        """ 保存FAISS向量库到本地 """
        if self.vector_store:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            self.vector_store.save_local(path)
            print(f"FAISS向量库已保存到：{path}")
        else:
            print("FAISS向量库为空！")

    def load_vector_store(self, path: str):
        """ 从本地加载FAISS向量库 """
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"FAISS向量库已从 {path} 加载。")
            return self.vector_store
        else:
            print("路径不存在！")
            return None

    def add_documents(self, documents: List[Document]):
        """ 向已存在的向量库中添加文档 """
        if self.vector_store is None:
            print("向量库为空，请先创建向量库。")
            return None

        chunks = self.split_documents(documents)
        self.vector_store.add_documents(chunks)
        print(f"已添加 {len(chunks)} 个文档块到向量库。")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """ 相似度搜索 """
        if self.vector_store is None:
            print("向量库为空，请先创建向量库。")
            return []
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """ 相似度搜索（带分数） """
        if self.vector_store is None:
            print("向量库为空，请先创建向量库。")
            return []
        return self.vector_store.similarity_search_with_score(query, k=k)

# ==================== 3. 检索模块 ====================

class Retriever:
    """ 检索器：从向量库中检索相关文档 """
    def __init__(self, vector_store: FAISS, config: RAGConfig):
        self.vector_store = vector_store
        self.config = config

    def retrieve(self, query: str) -> List[Document]:
        """ 检索相关文档 """
        return self.vector_store.similarity_search(
            query=query,
            k=self.config.top_k
        )

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """ 检索文档并返回相似度分数 """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=self.config.top_k
        )

# ==================== 4. 生成模块 ====================

class Generator:
    """ 生成器：基于上下文生成回答 """
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = model    # 这里用的是全局的model
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE.RAG_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{query}")
        ])
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE.REWRITE_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "原始问题：{query}\n\n请改写为独立完整的查询。")
        ])

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """ 根据对话历史改写查询 """
        if not chat_history:
            return query

        messages = []
        for msg in chat_history[-4:]:   # 这里暂且用后4条历史消息
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        chain = self.rewrite_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "chat_history": messages})

    def generate(self, query: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """ 生成回答 """
        messages = []
        if chat_history:
            for msg in chat_history[-4:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        chain = self.rag_prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "query": query,
            "context": context,
            "chat_history": messages
        })

    def evaluate_confidence(self, query: str, context: str, answer: str) -> float:
        """ 评估回答的置信度 """
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE.EVAL_PROMPT),
            ("human", """上下文：{context}
问题：{query}
回答：{answer}
置信度（0-1）：""")
        ])

        chain = eval_prompt | self.llm | StrOutputParser()
        try:
            score = float(chain.invoke({
                "context": context,
                "query": query,
                "answer": answer,
            }).strip())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5


# ==================== 5. RAG链整合 ====================

class RAGChain:
    """ RAG链：整合所有组件的完整流程 """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.processor = DocumentProcessor(self.config)
        self.retriever = None
        self.generator = Generator(self.config)
        self.graph = None

    def index_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """ 索引文档 """
        vector_store = self.processor.process(texts, metadatas)
        self.retriever = Retriever(vector_store, self.config)
        self._build_graph()

    def _build_graph(self):
        """ 构建Langgraph流程图 """

        def process_query(state: RAGState) -> RAGState:
            """ 改写查询（如有对话历史） """
            query = state["query"]
            chat_history = state.get("chat_history", [])

            if chat_history:
                rewritten = self.generator.rewrite_query(query, chat_history)
                print(f"查询改写： {query} -> {rewritten}")
                state["query"] = rewritten
            return state

        def retrieve_documents(state: RAGState) -> RAGState:
            """ 检索相关文档 """
            query = state["query"]

            docs = self.retriever.retrieve(query)
            print(f"检索到 {len(docs)} 个相关文档")
            state["documents"] = docs

            # 格式化上下文
            context_parts = []
            sources = []
            for i, doc in enumerate(docs):
                context_parts.append(f"[文档 {i+1}] {doc.page_content}")
                sources.append({
                    "index": i+1,
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:100] + "..."
                })
            state["context"] = "\n\n".join(context_parts)
            state["sources"] = sources

            return state

        def generate_answer(state: RAGState) -> RAGState:
            """ 生成回答 """
            answer = self.generator.generate(
                query=state["query"],
                context=state["context"],
                chat_history=state.get("chat_history", [])
            )
            state["answer"] = answer
            print("生成回答完成。")
            return state

        def evaluate_response(state: RAGState) -> RAGState:
            """ 评估回答置信度 """
            confidence = self.generator.evaluate_confidence(
                query=state["query"],
                context=state["context"],
                answer=state["answer"]
            )
            state["confidence"] = confidence
            print(f"置信度评估：{confidence:.2f}")
            return state

        # 构建图
        graph = StateGraph(RAGState)
        graph.add_node("process_query", process_query)
        graph.add_node("retrieve_documents", retrieve_documents)
        graph.add_node("generate_answer", generate_answer)
        graph.add_node("evaluate_response", evaluate_response)

        graph.add_edge(START, "process_query")
        graph.add_edge("process_query", "retrieve_documents")
        graph.add_edge("retrieve_documents", "generate_answer")
        graph.add_edge("generate_answer", "evaluate_response")
        graph.add_edge("evaluate_response", END)
        self.graph = graph.compile()

    def query(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """ 执行查询 """
        if not self.retriever:
            raise ValueError("请先调用 index_documents() 索引文档")

        print("=" * 50)
        print(f"提问：{question}")
        print("=" * 50)

        initial_state = {
            "query": question,
            "chat_history": chat_history or [],
            "documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"]
        }


# ==================== 6.示例数据与主程序 ====================

SAMPLE_DOCUMENTS = [
    {
        "text": """铁矿石-基本面

铁矿石是钢铁生产的重要原材料，其价格波动主要由供需关系、政策环境及下游需求共同决定。全球铁矿石供应高度集中，澳大利亚和巴西的四大矿商（淡水河谷、力拓、必和必拓、FMG）掌握着主要资源，发货量和天气因素（如飓风、雨季）会直接影响到港节奏。需求端与国内房地产、基建和制造业景气度紧密相连，粗钢产量变化是直接体现。钢厂在利润较高时倾向于采购高品矿以提升产量，利润微薄时则转向低品矿控制成本。库存方面，港口库存反映整体宽松程度，钢厂库存则体现补库意愿，通常节假日前会有一轮集中补库行情。此外，政策因素如国产矿复产推进、废钢替代效应以及钢材出口情况也会间接影响铁矿石价格。交易者需综合跟踪澳巴发运量、高炉开工率、港口库存变化以及螺纹钢利润等核心指标。""",
        "metadata": {"source": "iron_intro.txt", "topic": "iron"}
    },
    {
        "text": """螺纹钢-基本面

螺纹钢是建筑行业的核心钢材品种，广泛用于房地产、桥梁及城市基建项目。其价格运行具有明显的季节性特征，春季（3-4月）和秋季（9-10月）是传统施工旺季，需求集中释放；冬季则因北方停工导致需求萎缩，形成“北材南下”的贸易格局。供给端受制于钢铁企业的生产节奏、环保限产政策以及电炉开工率，其中唐山等主产区的限产文件往往引发供给收缩预期。需求端核心看房地产新开工面积和基建投资增速，同时货币政策和专项债发行节奏也会影响资金到位情况。库存数据是高频观测指标，社会库存与钢厂库存的“去库”速度直接反映当前供需强弱。成本支撑方面，铁矿石与焦炭价格变动会传导至螺纹钢生产成本。交易者需关注周度表观消费量、库存拐点以及地产销售数据，同时留意粗钢压减政策带来的中期影响。""",
        "metadata": {"source": "rebar_intro.txt", "topic": "rebar"}
    },
    {
        "text": """菜籽油-基本面

菜籽油是我国三大食用油之一，占植物油消费相当比重，具有浓郁的香味，在川渝等地区需求刚性较强。其价格波动受原料供应、压榨利润及替代油脂价差三重因素影响。供应端，国内菜籽产量有限，高度依赖进口，其中加拿大是我国最主要的菜籽和菜油来源国，因此中加贸易关系、海关政策以及进口到港量是核心变量。同时，菜籽压榨厂的开工率和压榨利润决定了阶段性供给，利润丰厚时油厂会加大压榨。需求端，菜籽油价格通常高于豆油和棕榈油，当价差过大时，食品工业和餐饮业会倾向于替代消费，从而抑制菜油需求。库存方面，华东和华南地区的菜油商业库存是重要参考，累库阶段价格承压，去库阶段支撑走强。此外，全球植物油市场的联动性很强，国际原油价格、棕榈油产量以及生物柴油政策都会通过比价效应传导至菜籽油市场。交易者需关注菜籽买船进度、油厂开机率以及三大油脂库存变化。""",
        "metadata": {"source": "rapeseed_oil_intro.txt", "topic": "rapeseed_oil"}
    },
    {
        "text": """甲醇-基本面

甲醇是重要的基础化工原料，广泛应用于烯烃、甲醛、醋酸和二甲醚等下游领域，其中甲醇制烯烃（MTO/MTP）占比最高，是需求端的核心支柱。国内甲醇生产以煤制路线为主，主要产地位于西北（内蒙古、陕西、宁夏）地区，成本受煤炭价格影响显著；进口则以天然气制为主，来源国主要是伊朗、阿曼等中东国家，进口到港量直接影响港口库存。由于产地与消费地分离，形成了特殊的贸易格局：西北产区主要供给周边下游，而华东、华南消费区则需要依赖进口和内地货源补充，区域价差和物流成本（汽运、火运、船运）决定套利窗口是否开启。需求端变化需重点关注MTO装置的开停工状态及利润水平，当聚烯烃利润不佳时，MTO装置可能检修或降负，直接减少甲醇消费。库存方面，港口库存尤其是华东沿海罐容水平是短期价格的敏感指标。同时，冬季天然气限气和春季煤化工检修会带来供给收缩预期。交易者应跟踪甲醇开工率、港口库存、MTO利润以及伊朗装船动态。""",
        "metadata": {"source": "methanol_intro.txt", "topic": "methanol"}
    }
]

def main():
    """ 主程序，演示RAG的使用 """
    print("=" * 50)
    # RAG配置
    print("初始化RAG系统")
    config = RAGConfig(
        chunk_size=100,
        chunk_overlap=20,
        top_k=3
    )
    rag = RAGChain(config)

    # 文档处理
    print("开始索引文档")
    texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
    metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]
    rag.index_documents(texts, metadatas)

    # # 单轮对话
    # print("测试单轮对话")
    # questions = [
    #     "螺纹钢生产成本受上游什么品种影响？",
    #     "我国的菜籽油主要从哪国进口？",
    #     "甲醇的下游领域有哪些？",
    # ]
    # for q in questions:
    #     answer = rag.query(q)
    #     print(f"\n提问：{q}\n回答：{answer}")

    # 多轮对话
    chat_history = []
    # 第一轮
    q1 = "我国什么地区的菜籽油需求刚性更强？"
    res1 = rag.query(q1, chat_history)
    print(f"\n提问：{q1}\n回答：{res1["answer"]}")
    chat_history.append({"role": "user", "content": q1})
    chat_history.append({"role": "assistant", "content": res1["answer"]})

    # 第二轮 指代消解
    q2 = "它和豆油通常谁的价格更高？"
    res2 = rag.query(q2, chat_history)
    print(f"\n提问：{q2}\n回答：{res2["answer"]}")
    chat_history.append({"role": "user", "content": q2})
    chat_history.append({"role": "assistant", "content": res2["answer"]})

    # 第三轮
    q3 = "交易者需要关注什么？"
    res3 = rag.query(q3, chat_history)
    print(f"\n提问：{q3}\n回答：{res3["answer"]}")
    chat_history.append({"role": "user", "content": q3})
    chat_history.append({"role": "assistant", "content": res3["answer"]})


if __name__ == "__main__":
    main()


