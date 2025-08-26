import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

if "API_KEY" not in st.session_state:
    st.session_state.API_KEY = None

with st.sidebar:
    st.title("ℹ️ Configuration")
    
    if not st.session_state.API_KEY:
        api_key = st.text_input(
            "Enter your API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from [AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.API_KEY = api_key
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.API_KEY = None
            st.rerun()
    
    st.info(
        "This tool provides AI-powered analysis of medical imaging data using "
        "advanced computer vision and radiological expertise."
    )
    st.warning(
        "⚠DISCLAIMER: This tool is for educational and informational purposes only. "
        "All analyses should be reviewed by qualified healthcare professionals. "
        "Do not make medical decisions based solely on this analysis."
    )

medical_agent = Agent(
    model=OpenAIChat(
        id="gpt-4o",
        api_key=st.session_state.API_KEY
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
) if st.session_state.API_KEY else None

if not medical_agent:
    st.warning("Please configure your API key in the sidebar to continue")

# Medical Analysis Query
query = """
        You are a highly skilled dermatology imaging expert with extensive knowledge in the analysis of skin conditions through various imaging modalologies. Analyze the patient's skin imaging and structure your response as follows:

        ### 1. Image Region
        - Identify the affected anatomical region and positioning of the lesion or area of interest
        - Comment on image quality, clarity, and technical adequacy for diagnosis

        ### 2. Key Findings
        - List primary observations systematically focusing on the lesion(s)
        - Note any abnormalities in the skin imaging with precise descriptions
        - Include measurements, color variations, textures, and borders where relevant
        - Describe location, size, shape, color, and other characteristics of the lesion(s)
        - Rate severity: Normal/Mild/Moderate/Severe

        ### 3. Diagnostic Assessment
        - Provide primary diagnosis with confidence level based on observed evidence
        - List differential diagnoses in order of likelihood considering similar skin conditions
        - Support each diagnosis with observed evidence from the patient's imaging
        - Note any critical or urgent findings that require immediate attention

        ### 4. Patient-Friendly Explanation
        - Explain the findings in simple, clear language that the patient can understand
        - Avoid medical jargon or provide clear definitions when necessary
        - Use visual analogies to help explain the condition if helpful
        - Address common patient concerns related to these findings and possible outcomes

        ### 5. Research Context
        IMPORTANT: Use the DuckDuckGo search tool to:
        - Find recent medical literature about similar cases involving the diagnosed condition
        - Search for standard treatment protocols and guidelines for the condition
        - Provide a list of relevant medical links and resources
        - Research any relevant technological advances in diagnosing or treating the condition
        - Include 2-3 key references to support your analysis and recommendations

        Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = PILImage.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="Uploaded Medical Image",
                use_container_width=True
            )
            
            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            )
    
    with analysis_container:
        if analyze_button:
            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    temp_path = "temp_resized_image.png"
                    resized_image.save(temp_path)
                    
                    # Create AgnoImage object
                    agno_image = AgnoImage(filepath=temp_path)  # Adjust if constructor differs
                    
                    # Run analysis
                    response = medical_agent.run(query, images=[agno_image])
                    st.markdown("### 📋 Analysis Results")
                    st.markdown("---")
                    st.markdown(response.content)
                    st.markdown("---")
                    st.caption(
                        "Note: This analysis is generated by AI and should be reviewed by "
                        "a qualified healthcare professional."
                    )
                except Exception as e:
                    st.error(f"Analysis error: {e}")
else:
    st.info("👆 Please upload a medical image to begin analysis")
