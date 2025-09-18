import os
import torch
import asyncio
from llama_index.core import Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from agno.agent import Agent
from agno.knowledge.llamaindex import LlamaIndexKnowledgeBase
from agno.media import Image as AgnoImage
from agno.models.google.gemini import Gemini
from utils import process_markdown
from prompt_template import get_domain_expert_prompt, get_rag_prompt
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)
# 让 llama-index 组件也输出
logging.getLogger("llama_index").setLevel(logging.INFO)
from llama_index.core.schema import NodeWithScore, QueryBundle  # 仅用于类型提示


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
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            device=device
        )

        # Initialize vector stores
        self.text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.text_store
        )

        # Build the agent during initialization
        self.agent = self._build_agent()

    def retrieve_knowledge(self, query_text: str, top_k: int = 3):
        """用任意文本去知识库检索，返回最相关的知识片段"""
        retriever = self.agent.knowledge.retriever
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
        self.text_store.add(nodes)
        count = self.text_store._table.count_rows()  # ✅ 正确计数
        print(f"【建索引导入完成】共写入 {count} 条向量")
        # Create MultiModalVectorStoreIndex
        vector_index = MultiModalVectorStoreIndex.from_vector_store(self.text_store)

        # Set up retriever and knowledge base
        retriever = vector_index.as_retriever()
        knowledge_base = LlamaIndexKnowledgeBase(retriever=retriever)

        # Initialize agent
        agent = Agent(
            model=Gemini(id=self.model, api_key=self.api_key),
            knowledge=knowledge_base,
            search_knowledge=True,
            debug_mode=False,
            show_tool_calls=True
        )

        return agent

    async def analyze(self, query, image_file_path):
        """
        Analyze an image using the configured agent.

        Args:
            query (str): The query to run on the image.
            image_file_path (str): Path to the image file for analysis.

        Returns:
            str: The analysis result.
        """
        # Load image
        agno_image = AgnoImage(filepath=image_file_path)
        # Run analysis
        # Use asyncio.to_thread to run the synchronous method in a separate thread
        response = await asyncio.to_thread(self.agent.run, query, images=[agno_image])
        return response.content


if __name__ == "__main__":
    # Build the agent
    image_file_path = "./SkinGPT-X-Dataset/Dermnet/test/Seborrheic Keratoses and other Benign Tumors/seborrheic-keratosis-irritated-28.jpg"  # Replace with your actual image file path
    model = "gemini-2.5-pro"
    api_key = "AIzaSyC-9og_9OsxvKZ0rBXeMGboXBrMOpG5-do"
    markdown_file_path = "./skin_handbook.md"
    domain = "RAG"
    builder = RAGAgent(model=model, api_key=api_key, domain=domain, markdown_file_path=markdown_file_path)


    async def main():
        query = get_domain_expert_prompt('RAG')
        analysis_result = await builder.analyze(query, image_file_path)
        print(analysis_result)
    asyncio.run(main())