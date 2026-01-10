import os
import asyncio
from llama_index.core import Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from agno.agent import Agent
from agno.knowledge.llamaindex import LlamaIndexKnowledgeBase
from agno.media import Image as AgnoImage
from agno.models.google.gemini import Gemini
from utils import process_markdown, encode_image_to_base64
from prompt_template import get_domain_expert_prompt, get_rag_prompt
import logging, sys
from llama_index.core.schema import NodeWithScore, QueryBundle  # 仅用于类型提示
from agno.models.openai import OpenAIChat
import re
import ssl
import json
from local_llm_utils import local_generate_response_vl, parse_skin_disease_path
from Constants import DERMNET_DISEASE_NAME

# 全局禁用 SSL 验证（影响整个 Python 进程）
ssl._create_default_https_context = ssl._create_unverified_context
class RAGAgent:
    def __init__(self, model, api_key, domain, markdown_file_path, use_gpu=False):
        """
        Initialize the Medical Agent Builder with an API key, domain, and markdown file path.

        Args:
            model (str): The model to use for the agent.
            api_key (str): The API key for accessing external services.
            domain (str): The domain of the agent (e.g., "MedicalImaging").
            markdown_file_path (str): Path to the markdown file containing knowledge.
        """
        self.model = model
        self.api_key = api_key
        self.domain = domain
        self.markdown_file_path = markdown_file_path
        os.environ["GEMINI_APIKEY"] = self.api_key

        # ★ 根据开关决定设备

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="/225040511/project/hf_cache/bge-small-en-v1.5/BAAI/bge-small-en-v1___5")

        # Initialize vector stores
        # LOCAL_DB = "/content/lancedb"  # ← 本地可写路径
        # os.makedirs(LOCAL_DB, exist_ok=True)
        # self.text_store = LanceDBVectorStore(uri=LOCAL_DB, table_name="text_collection", mode="create")
        self.text_store = LanceDBVectorStore(uri="/225040511/project/Skingpt_X/lancedb", table_name="text_collection", mode="ro")
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.text_store
        )

        # Build the agent during initialization
        self.knowledge_base = self._build_agent()

    def retrieve_knowledge(self, query_text: str, top_k: int = 3):
        """用任意文本去知识库检索，返回最相关的知识片段"""
        retriever = self.knowledge_base.retriever
        nodes = retriever.retrieve(QueryBundle(query_text))
        return "\n\n".join([n.node.text for n in nodes[:top_k]])

    def _build_agent(self):
        """
        Build the medical agent by processing markdown files and setting up the knowledge base.

        Returns:
            Agent: The configured agent instance.
        """
        # Process markdown file to extract nodes
        nodes = process_markdown(self.markdown_file_path)
        embed_model = Settings.embed_model
        for node in nodes:
            node.embedding = embed_model.get_text_embedding(node.text)
        # self.text_store.add(nodes)
        count = self.text_store._table.count_rows()  # ✅ 正确计数
        print(f"【建索引导入完成】共写入 {count} 条向量")
        # Create MultiModalVectorStoreIndex
        vector_index = VectorStoreIndex.from_vector_store(self.text_store)

        # Set up retriever and knowledge base
        retriever = vector_index.as_retriever()
        knowledge_base = LlamaIndexKnowledgeBase(retriever=retriever)
        return knowledge_base

    def analyze(self, image_path, mode='test', ground_truth=None, pred_name=None, prob_value=None):
        """
        mode: 'test' (推理模式), 'train_exercise' (训练-盲测模式), 'train_gold' (训练-标杆模式)
        """
        # 1) 获取 RAG 上下文 (Handbook 知识)
        context = self.retrieve_knowledge(pred_name, top_k=10)

        # 2) 根据模式选择不同的 Prompt 策略
        if mode == 'train_exercise':
            # 实战演练：给它 Panderm 的预测，看它会不会被带偏
            prompt = get_rag_prompt(pred_name, prob_value, context, MAPPINGSET=DERMNET_DISEASE_NAME)
        elif mode == 'train_gold':
            # 标准归档：直接给它 Ground Truth，要求生成最完美的逻辑描述
            prompt = get_rag_prompt_with_true_label(ground_truth, context)
        else:
            # 正常的测试模式
            prompt = get_rag_prompt(pred_name, prob_value, context, MAPPINGSET=DERMNET_DISEASE_NAME)

        # 3) 调用 VL 模型
        print(prompt)
        response = local_generate_response_vl(
                        temperature=0.2,
                        max_tokens=4096,
                        prompt=prompt,
                        image_path=image_path
                    )
        # 4) 解析 JSON
        try:
            # 转换为 Python字典
            result = json.loads(response)
                    
                
            return result
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return {"error": "parse_error", "raw": response}


if __name__ == "__main__":
    # Build the agent
    image_file_path = "/225040511/Dataset/HAM10000/test/ISIC_0034525.jpg"  # Replace with your actual image file path
    model = "gemini-2.5-pro"
    api_key = "sk-iCv69YeaJn8TXm9tk6ZUUAqftw51aB2yddvmstNNl7QjkIKB"
    markdown_file_path = "./skin_handbook.md"
    domain = "RAG"
    builder = RAGAgent(model=model, api_key=api_key, domain=domain, markdown_file_path=markdown_file_path)
    query = {"Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
              "basal cell carcinoma",
              "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)",
              "dermatofibroma", "melanoma", "melanocytic nevi ",
              "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)"}
    
    async def main():
        analysis_result = await builder.analyze(query, image_file_path)
        print(analysis_result)
    asyncio.run(main())