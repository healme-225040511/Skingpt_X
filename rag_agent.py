import os
from llama_index.core import Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from agno.agent import Agent
from agno.knowledge.llamaindex import LlamaIndexKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.media import Image as AgnoImage
from utils import process_markdown
from prompt_template import *


class RAGAgent:
    def __init__(self, model, api_key, domain, markdown_file_path):
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
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Initialize vector stores
        self.text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.text_store
        )

        # Build the agent during initialization
        self.agent = self._build_agent()

    def _build_agent(self):
        """
        Build the medical agent by processing markdown files and setting up the knowledge base.
        
        Returns:
            Agent: The configured agent instance.
        """
        # Process markdown file to extract nodes
        nodes = process_markdown(self.markdown_file_path)

        # Create MultiModalVectorStoreIndex
        vector_index = MultiModalVectorStoreIndex.from_vector_store(self.text_store)

        # Set up retriever and knowledge base
        retriever = vector_index.as_retriever()
        knowledge_base = LlamaIndexKnowledgeBase(retriever=retriever)

        # Initialize agent
        agent = Agent(
            model=OpenAIChat(id=self.model, api_key=self.api_key),
            knowledge=knowledge_base, 
            search_knowledge=True,
            debug_mode=False, 
            show_tool_calls=True
        )

        return agent

    def analyze(self, query, image_file_path):
        """
        Analyze an image using the configured agent.
        
        Args:
            agent (Agent): The configured agent instance.
            query (str): The query to run on the image.
            image_file_path (str): Path to the image file for analysis.
            
        Returns:
            str: The analysis result.
        """
        # Load image
        agno_image = AgnoImage(filepath=image_file_path)

        # Run analysis
        response = self.agent.run(query, images=[agno_image])
        return response.content


if __name__ == "__main__":
    # Build the agent
    model = "gpt-4o-mini"
    api_key = ""
    markdown_file_path = "D:\\Projects\\skinGPT4x\\awesome-llm-apps\\ai_agent_tutorials\\ai_medical_imaging_agent\\skin_handbook.md"
    domain = "RAG"
    builder = RAGAgent(model=model, api_key=api_key, domain=domain, markdown_file_path=markdown_file_path)

    # Run analysis
    image_file_path = "D:\\Projects\\skinGPT4x\\skinCAD\\SkinCAP\\skincap\\167.png"
    query = get_domain_expert_prompt(domain)
    analysis_result = builder.analyze(query, image_file_path)
    print(analysis_result)