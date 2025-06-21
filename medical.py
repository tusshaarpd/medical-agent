import os
import sys
import streamlit as st
from dotenv import load_dotenv
import base64
from io import BytesIO

# 1. MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="🩺 Medical Report Analyzer", layout="wide")

# 2. Load environment variables first
load_dotenv()

# 3. Import required libraries
try:
    from PIL import Image
    import io
    import PyPDF2
    import fitz  # PyMuPDF for better PDF handling
    
    # Import CrewAI components
    from crewai import Agent, Task, Crew, Process
    from langchain_openai import ChatOpenAI
    
except ImportError as e:
    st.error(f"Missing required packages: {e}")
    st.info("Please install: pip install crewai langchain-openai pillow PyPDF2 PyMuPDF")
    st.stop()

# Initialize OpenAI LLM via API
try:
    llm = ChatOpenAI(
        model="gpt-4o",  # Using GPT-4 with vision capabilities
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    st.error(f"Failed to initialize OpenAI LLM: {e}")
    st.stop()

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        # Try with PyMuPDF first (better for complex PDFs)
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
            
        pdf_document.close()
        return text.strip()
        
    except Exception as e:
        # Fallback to PyPDF2
        try:
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            return text.strip()
            
        except Exception as e2:
            st.error(f"Failed to extract text from PDF: {str(e2)}")
            return None

def encode_image_to_base64(image):
    """Convert PIL image to base64 string for API"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_image_with_gpt4_vision(image):
    """Analyze image using GPT-4 Vision API"""
    try:
        # Convert image to base64
        img_base64 = encode_image_to_base64(image)
        
        # Create vision-enabled LLM
        vision_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt for medical image analysis
        prompt = f"""
        Please analyze this medical report image and provide a detailed interpretation. 
        
        Focus on:
        1. Extracting all visible text and data
        2. Identifying medical findings, test results, or abnormalities
        3. Explaining the significance of any measurements or values
        4. Noting any diagnostic information present
        
        Provide a comprehensive analysis that captures all the medical information visible in the image.
        
        Image: data:image/png;base64,{img_base64}
        """
        
        response = vision_llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        st.error(f"Error during image analysis: {str(e)}")
        return None

# Define medical analysis agents
def create_agents():
    analyzer_agent = Agent(
        role="Medical Report Analyzer",
        goal="Analyze uploaded medical reports and summarize findings in simple terms",
        backstory="A compassionate medical expert trained to explain clinical findings in layman's language.",
        llm=llm,
        verbose=True
    )

    cause_agent = Agent(
        role="Cause Identifier", 
        goal="Identify potential causes of abnormalities found in the report",
        backstory="A diagnostician who hypothesizes underlying causes based on medical data.",
        llm=llm,
        verbose=True
    )

    remedy_agent = Agent(
        role="Remedy Recommender",
        goal="Suggest possible lifestyle changes or treatments for abnormalities found in the report", 
        backstory="An advisor helping people take initial steps based on report findings.",
        llm=llm,
        verbose=True
    )
    
    return analyzer_agent, cause_agent, remedy_agent

# Streamlit UI
st.title("🩺 Medical Report Analyzer")
st.markdown("Upload your medical reports (PDF or images) for AI-powered analysis and insights.")

# Check if required environment variables are set
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Please set your OPENAI_API_KEY environment variable")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Upload your medical report", 
    type=["jpg", "jpeg", "png", "pdf"],
    help="Supported formats: JPG, PNG, PDF"
)

if uploaded_file and st.button("🔍 Analyze Report"):
    try:
        file_type = uploaded_file.type
        extracted_text = None
        
        with st.spinner("Processing your medical report..."):
            
            if file_type == "application/pdf":
                # Handle PDF files
                st.info("📄 Processing PDF file...")
                extracted_text = extract_text_from_pdf(uploaded_file)
                
                if extracted_text:
                    st.success("✅ Successfully extracted text from PDF")
                    with st.expander("📝 Extracted Text Preview", expanded=False):
                        st.text_area("", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=200)
                else:
                    st.error("❌ Could not extract text from PDF. The PDF might be image-based or corrupted.")
                    st.stop()
                    
            else:
                # Handle image files
                st.info("🖼️ Processing image file...")
                image = Image.open(uploaded_file)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Medical Report", use_column_width=True)
                
                # Analyze the image using GPT-4 Vision
                extracted_text = analyze_image_with_gpt4_vision(image)
                
                if extracted_text:
                    st.success("✅ Successfully analyzed image")
                    with st.expander("🔍 Image Analysis", expanded=False):
                        st.write(extracted_text)
                else:
                    st.error("❌ Could not analyze the image. Please ensure it's clear and contains medical information.")
                    st.stop()

        # Proceed with comprehensive analysis if we have extracted text
        if extracted_text and extracted_text.strip():
            
            # Create agents
            analyzer_agent, cause_agent, remedy_agent = create_agents()
            
            # Define tasks based on extracted text
            analyze_task = Task(
                description=f"Take this medical report content and create a clear, patient-friendly summary:\n\n{extracted_text}\n\nFocus on explaining any medical findings, test results, or abnormalities mentioned in terms that a patient can easily understand. Highlight key values, measurements, and their significance.",
                expected_output="A clear, patient-friendly summary of the medical findings with explanations of medical terms and their significance.",
                agent=analyzer_agent
            )

            cause_task = Task(
                description=f"Based on the medical findings from this report: '{extracted_text}', identify and explain the most likely medical causes or factors that could lead to these findings. Focus on common, well-established medical knowledge and provide educational context.",
                expected_output="Clear explanation of potential medical causes for the findings, written in accessible language with educational context.",
                agent=cause_agent
            )

            remedy_task = Task(
                description=f"Based on this medical analysis: '{extracted_text}', provide practical, evidence-based health recommendations. Include lifestyle modifications, when to seek medical care, preventive measures, and general wellness advice that could be beneficial. Be specific and actionable.",
                expected_output="Comprehensive, actionable health recommendations, lifestyle advice, and clear guidance on next steps for medical care.",
                agent=remedy_agent
            )

            # Create and run Crew
            with st.spinner("🤖 Running comprehensive medical analysis with AI agents..."):
                try:
                    medical_crew = Crew(
                        agents=[analyzer_agent, cause_agent, remedy_agent],
                        tasks=[analyze_task, cause_task, remedy_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    result = medical_crew.kickoff()
                    
                    # Display results in organized sections
                    st.success("✅ Analysis Complete!")
                    
                    # Main comprehensive analysis
                    st.subheader("📋 Comprehensive Medical Analysis")
                    
                    # Handle different result formats
                    if hasattr(result, 'raw'):
                        st.markdown(result.raw)
                    elif hasattr(result, 'tasks_output') and result.tasks_output:
                        # Display the final task output
                        final_output = result.tasks_output[-1]
                        if hasattr(final_output, 'raw'):
                            st.markdown(final_output.raw)
                        else:
                            st.markdown(str(final_output))
                    else:
                        st.markdown(str(result))
                    
                    # Show individual agent outputs in organized tabs
                    if hasattr(result, 'tasks_output') and result.tasks_output:
                        tab1, tab2, tab3 = st.tabs(["📊 Summary", "🔬 Causes", "💊 Recommendations"])
                        
                        with tab1:
                            st.markdown("### Medical Report Summary")
                            if hasattr(result.tasks_output[0], 'raw'):
                                st.markdown(result.tasks_output[0].raw)
                            else:
                                st.markdown(str(result.tasks_output[0]))
                        
                        with tab2:
                            st.markdown("### Potential Causes Analysis")
                            if len(result.tasks_output) > 1:
                                if hasattr(result.tasks_output[1], 'raw'):
                                    st.markdown(result.tasks_output[1].raw)
                                else:
                                    st.markdown(str(result.tasks_output[1]))
                        
                        with tab3:
                            st.markdown("### Health Recommendations")
                            if len(result.tasks_output) > 2:
                                if hasattr(result.tasks_output[2], 'raw'):
                                    st.markdown(result.tasks_output[2].raw)
                                else:
                                    st.markdown(str(result.tasks_output[2]))

                except Exception as crew_error:
                    st.error(f"CrewAI analysis failed: {str(crew_error)}")
                    st.info("🔄 Attempting direct analysis...")
                    
                    # Fallback analysis
                    prompt = f"""
                    As a medical expert, analyze this medical report content:

                    {extracted_text}

                    Provide a comprehensive analysis with:

                    ## 📊 PATIENT-FRIENDLY SUMMARY
                    Explain the key findings in simple terms that a patient can understand. Include specific values and their normal ranges where applicable.

                    ## 🔬 POSSIBLE CAUSES  
                    What medical conditions or factors could explain these findings? Provide educational context about these conditions.

                    ## 💊 RECOMMENDATIONS
                    What practical steps, lifestyle changes, or medical follow-up would you recommend? Be specific and actionable.

                    ## ⚡ NEXT STEPS
                    What should the patient do next? When should they see a doctor? Any urgent concerns?

                    Use clear, accessible language and organize the response with headers as shown.
                    """
                    
                    try:
                        response = llm.invoke(prompt)
                        st.subheader("📋 Medical Analysis")
                        st.markdown(response.content)
                    except Exception as llm_error:
                        st.error(f"Direct analysis also failed: {str(llm_error)}")

            # Medical disclaimer
            st.markdown("---")
            st.error("""
            ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
            
            This AI analysis is provided for **informational and educational purposes only**. 
            
            **This is NOT medical advice and should never replace:**
            - Professional medical consultation
            - Clinical diagnosis by qualified healthcare providers  
            - Treatment recommendations from your doctor
            - Emergency medical care when needed
            
            **Always consult qualified healthcare professionals for any medical concerns or before making health-related decisions.**
            
            **For emergencies, contact your local emergency services immediately.**
            """)
            
        else:
            st.error("❌ Could not extract meaningful information from your file.")
            st.info("Please ensure your file:")
            st.write("- Contains readable medical text or clear images")
            st.write("- Is not corrupted or password-protected")
            st.write("- Has good quality and resolution (for images)")
            st.write("- Contains actual medical report content")
            
    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        st.info("Please try again with a different file or check your setup.")

# Sidebar information
with st.sidebar:
    st.header("🤖 AI-Powered Analysis")
    st.write("""
    This tool uses **GPT-4** with vision capabilities and **CrewAI** multi-agent system to provide comprehensive medical report analysis.
    """)
    
    st.header("📁 Supported File Types")
    st.write("""
    **Images:**
    - JPG, JPEG, PNG
    - Clear, high-resolution preferred
    
    **Documents:**
    - PDF files with text or images
    - Multi-page reports supported
    """)
    
    st.header("🔧 Setup Requirements")
    st.write("""
    **Required Environment Variables:**
    - `OPENAI_API_KEY`: OpenAI API key for GPT-4 access
    
    **Recommended:**
    - Stable internet connection
    - Clear, readable report files
    """)
    
    st.header("💡 Tips for Best Results")
    st.write("""
    - Upload clear, high-resolution images
    - Ensure text in reports is legible
    - Include complete reports when possible
    - Use original files (not screenshots when possible)
    - Make sure PDFs are not password-protected
    """)
    
    st.header("🏥 What We Analyze")
    st.write("""
    - Blood tests and lab results
    - Imaging reports (X-ray, MRI, CT, etc.)
    - Pathology reports
    - Diagnostic summaries
    - Health checkup reports
    - And more medical documents
    """)

st.markdown("---")
st.markdown("*Powered by GPT-4 Vision + CrewAI Multi-Agent System*")

# Add some usage examples
with st.expander("📖 How to Use This Tool"):
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Upload Your Report**: Click the upload button and select your medical report (PDF or image)
    
    2. **Wait for Processing**: The AI will extract and analyze the content
    
    3. **Review Analysis**: Get a comprehensive breakdown including:
       - Patient-friendly summary of findings
       - Potential causes of any abnormalities
       - Actionable health recommendations
       - Next steps and follow-up guidance
    
    4. **Consult Your Doctor**: Use this analysis as a starting point for discussions with your healthcare provider
    
    ### Best Practices:
    - Always verify AI insights with qualified medical professionals
    - Use this tool for educational purposes and preparation for doctor visits
    - Keep your original reports for medical consultations
    - Don't delay seeking medical care based on AI analysis alone
    """)
