import streamlit as st
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import requests
import json
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="PhospheneAI Research Assistant", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# Beautiful CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-title {
        text-align: center;
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin-bottom: 3rem;
    }
    
    .response-box {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border-left: 6px solid #4CAF50;
    }
    
    .ollama-status {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
    }
    
    .source-card {
        background: linear-gradient(135deg, #FF6B6B, #FF8E8E);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .stTextInput > div > div > input {
        border-radius: 50px !important;
        border: 3px solid white !important;
        padding: 1rem 2rem !important;
        font-size: 1.2rem !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        color: white;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Load FAISS database
@st.cache_resource
def load_resources():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed(texts):
            return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        db = FAISS.load_local("faiss_index", embeddings=embed, allow_dangerous_deserialization=True)
        return model, db
    except Exception as e:
        st.error(f"❌ Error loading database: {str(e)}")
        return None, None

model, db = load_resources()

# Ollama integration functions
def check_ollama_status():
    """Check if Ollama is running and get available models"""
    try:
        response = requests.get("http://10.10.50.198:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {"status": "running", "models": [m["name"] for m in models]}
        else:
            return {"status": "error", "message": "Ollama not responding"}
    except requests.exceptions.ConnectionError:
        return {"status": "not_running", "message": "Ollama not started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def call_ollama(prompt, model_name="llama2:7b", context=""):
    """Optimized Ollama API call with timeout and performance fixes"""
    try:
        # Shorter, more efficient prompt
        full_prompt = f"""Based on this news context, answer the question clearly and concisely:

Context: {context[:1000]}

Question: {prompt}

Answer:"""

        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 400,  # Shorter responses = faster
                "num_ctx": 2048,     # Smaller context window
                "top_p": 0.9
            },
            "keep_alive": "5m"  # Keep model loaded for 5 minutes
        }
        
        # Much longer timeout + progress indicator
        response = requests.post(
            "http://10.10.50.198:11434/api/generate",
            json=payload,
            timeout=600  # 10 minutes - should be enough
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "response": result.get("response", ""),
                "model": model_name,
                "done": result.get("done", True)
            }
        else:
            return {
                "success": False,
                "error": f"Ollama API error: {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Model is taking too long. Try a smaller model like 'gemma:2b'"}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}

# Check Ollama status
ollama_info = check_ollama_status()

# Main UI
st.markdown('<h1 class="main-title">🧠 PhospheneAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Local AI (Ollama) + Vector Search</p>', unsafe_allow_html=True)

# Ollama status indicator
if ollama_info["status"] == "running":
    available_models = ollama_info.get("models", [])
    if available_models:
        st.markdown(f"""
        <div class="ollama-status">
            ✅ Ollama is running with {len(available_models)} model(s): {', '.join(available_models)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠ Ollama is running but no models found. Download a model first:")
        st.code("ollama pull llama2:7b")
        st.stop()
elif ollama_info["status"] == "not_running":
    st.error("❌ Ollama not running. Please start Ollama first.")
    st.stop()
else:
    st.warning(f"⚠ Ollama issue: {ollama_info.get('message', 'Unknown error')}")

# Sidebar settings
with st.sidebar:
    st.markdown("### ⚙ AI Configuration")
    
    # Model selection
    available_models = ollama_info.get("models", ["llama2:7b"])
    if available_models:
        selected_model = st.selectbox(
            "🤖 Select AI Model",
            options=available_models,
            index=0
        )
    else:
        st.error("No models available. Run: ollama pull llama2:7b")
        selected_model = "llama2:7b"
    
    # Search settings
    search_count = st.slider("📚 Articles to analyze", 3, 8, 5)
    use_ai = st.toggle("🤖 Use AI Enhancement", value=True)
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📊 System Status")
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 AI</h4>
            <p>Online</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status2:
        db_status = "Online" if db else "Offline"
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔍 Database</h4>
            <p>{db_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Actions")
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()
    
    if st.button("📊 Show Stats", use_container_width=True):
        if st.session_state.conversation:
            total_questions = len(st.session_state.conversation)
            st.success(f"📈 {total_questions} questions answered")
        else:
            st.info("No conversations yet")

# Search interface
query = st.text_input(
    "",
    placeholder="🔍 Ask me anything about the news... (e.g., 'Latest developments in artificial intelligence')",
    key="search_query",
    disabled=st.session_state.processing
)

# Process query
if query and not st.session_state.processing:
    if not db:
        st.error("❌ News database not available. Check FAISS index.")
        st.stop()
    
    st.session_state.processing = True
    
    try:
        with st.spinner("🔍 Searching news database..."):
            # Search articles
            results = db.similarity_search(query, k=search_count)
            
            if not results:
                st.warning("⚠ No relevant articles found. Try different keywords.")
                st.session_state.processing = False
                st.stop()
            
            # Extract context and sources
            context = "\n\n".join([doc.page_content for doc in results])
            sources = [doc.metadata.get("source", f"Article {i+1}") for i, doc in enumerate(results)]
        
        # Create response layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if use_ai and ollama_info["status"] == "running" and available_models:
                with st.spinner(f"🤖 Generating AI response with {selected_model}..."):
                    ai_result = call_ollama(query, selected_model, context)
                    
                    if ai_result["success"]:
                        # Clean and format the response
                        ai_response = ai_result["response"].strip()
                        
                        st.markdown(f"""
                        <div class="response-box">
                            <h3>🤖 AI-Enhanced Response</h3>
                            <div style="line-height: 1.8; color: #333; font-size: 1.1rem;">
                                {ai_response.replace(chr(10), '<br>').replace('', '<strong>').replace('', '</strong>')}
                            </div>
                            <hr style="margin: 1.5rem 0; opacity: 0.3;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <small style="color: #666;">
                                    <strong>🤖 Model:</strong> {ai_result["model"]} | 
                                    <strong>🕒 Generated:</strong> {datetime.now().strftime('%H:%M:%S')}
                                </small>
                                <small style="color: #28a745; font-weight: 600;">✨ AI-Powered</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Save to conversation history
                        st.session_state.conversation.append({
                            "question": query,
                            "response": ai_response,
                            "model": selected_model,
                            "sources": sources,
                            "timestamp": datetime.now()
                        })
                        
                    else:
                        st.error(f"❌ AI Error: {ai_result['error']}")
                        # Fallback to context display
                        st.markdown(f"""
                        <div class="response-box">
                            <h3>📄 Retrieved Information</h3>
                            <p style="line-height: 1.6; color: #333;">{context[:1500]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Display retrieved context only
                st.markdown(f"""
                <div class="response-box">
                    <h3>📄 Retrieved Information</h3>
                    <p style="line-height: 1.6; color: #333;">{context[:1500]}...</p>
                    <hr style="margin: 1.5rem 0; opacity: 0.3;">
                    <small style="color: #666;">💡 Enable AI Enhancement for intelligent responses</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📚 Source Articles")
            for i, (doc, source) in enumerate(zip(results[:4], sources[:4]), 1):
                relevance_stars = "⭐" * (5 - min(i-1, 4))
                st.markdown(f"""
                <div class="source-card">
                    <h4>📰 Source {i}</h4>
                    <p><strong>From:</strong> {source}</p>
                    <p><strong>Relevance:</strong> {relevance_stars}</p>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.4;">
                        {doc.page_content[:180]}...
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Action buttons (only show if we have conversation history)
        if st.session_state.conversation:
            st.markdown("---")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                last_conversation = st.session_state.conversation[-1]
                report_content = f"""# PhospheneAI Research Report

*Question:* {last_conversation['question']}
*Generated:* {last_conversation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
*AI Model:* {last_conversation['model']}

## AI Response:
{last_conversation['response']}

## Sources Analyzed:
{chr(10).join([f"{i+1}. {source}" for i, source in enumerate(last_conversation['sources'])])}

---
Generated by PhospheneAI Research Assistant
Powered by Ollama + FAISS Vector Search
"""
                st.download_button(
                    "💾 Download Report",
                    report_content,
                    f"phospheneai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown",
                    use_container_width=True
                )
            
            with col_b:
                if st.button("📜 Show History", use_container_width=True):
                    with st.expander("💬 Conversation History", expanded=True):
                        for i, conv in enumerate(reversed(st.session_state.conversation), 1):
                            st.markdown(f"{i}. {conv['question']}")
                            st.markdown(f"{conv['timestamp'].strftime('%H:%M:%S')} - {conv['model']}")
                            st.markdown(f"{conv['response'][:250]}...")
                            st.divider()
            
            with col_c:
                if st.button("🔗 Share Results", use_container_width=True):
                    st.success("✅ Results ready to share!")
                    st.balloons()
            
            with col_d:
                if st.button("🆕 New Search", use_container_width=True):
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
    
    finally:
        st.session_state.processing = False

elif query and not available_models:
    st.error("❌ No AI models available. Please install a model:")
    st.code("ollama pull llama2:7b")
