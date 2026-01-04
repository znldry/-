# app.py
import streamlit as st
import sys
import os
import time

# 将项目根目录和src目录加入路径，确保模块导入正常
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# 设置页面配置（必须放在所有Streamlit命令之前）
st.set_page_config(
    page_title="医疗RAG问答系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题和描述
st.title("🏥 医疗检索增强生成 (RAG) 系统")
st.markdown("""
    基于 **Milvus** 向量数据库与 **Qwen** 大语言模型构建。
    系统首先从医学知识库中检索相关信息，然后生成准确、可靠的答案。
""")
st.divider()

# 在侧边栏进行系统设置和状态检查
with st.sidebar:
    st.header("⚙️ 系统控制")
    
    # 检索参数设置
    top_k = st.slider("检索返回结果数量 (top_k)", min_value=1, max_value=10, value=4, help="影响答案的参考信息广度")
    temperature = st.slider("生成温度 (temperature)", min_value=0.1, max_value=1.5, value=0.7, step=0.1, help="值越高，答案创造性越强，但可能更不稳定")
    
    st.divider()
    st.header("📊 系统状态")
    
    # 初始化关键组件到session_state（避免重复加载）
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
        st.session_state.vector_store = None
        st.session_state.generator = None
        st.info("系统未初始化")
    else:
        st.success("✅ 系统已就绪")
    
    # 初始化按钮
    if st.button("🔄 初始化/重启系统", type="primary", use_container_width=True):
        with st.spinner("正在加载系统组件，这可能需要几分钟..."):
            try:
                # 动态导入，避免启动时即加载
                from vector_store import MedicalVectorStore
                from generator import MedicalRAGGenerator
                from pymilvus import Collection, utility
                
                # 1. 初始化向量存储
                st.session_state.vector_store = MedicalVectorStore(collection_name="medical_knowledge_v1")
                if not st.session_state.vector_store.connect_to_milvus():
                    st.error("❌ 连接向量数据库失败，请确保Milvus服务正在运行。")
                    st.stop()
                
                # 检查并加载集合
                if not utility.has_collection(st.session_state.vector_store.collection_name):
                    st.error(f"集合不存在，请先运行 `vector_store.py` 导入数据。")
                    st.stop()
                st.session_state.vector_store.collection = Collection(st.session_state.vector_store.collection_name)
                st.session_state.vector_store.collection.load()
                
                # 2. 初始化生成器（此步骤较慢）
                with st.status("正在加载生成模型...", expanded=True) as status:
                    st.write("下载并初始化Qwen模型（首次运行需下载约3GB数据）...")
                    st.session_state.generator = MedicalRAGGenerator(retriever=st.session_state.vector_store)
                    status.update(label="模型加载成功！", state="complete")
                
                st.session_state.rag_initialized = True
                st.success("系统初始化完成！")
                time.sleep(1)
                st.rerun() # 刷新页面
                
            except Exception as e:
                st.error(f"初始化失败: {e}")
                st.session_state.rag_initialized = False

# 主界面区域
if not st.session_state.rag_initialized:
    st.warning("👈 请先在左侧边栏点击 **'初始化/重启系统'** 按钮来启动系统。")
    st.info("""
        **初始化步骤说明:**
        1. 确保 `standalone.bat start` 窗口正在运行（Milvus服务）。
        2. 点击左侧的初始化按钮。
        3. 首次加载模型需要较长时间和稳定的网络，请耐心等待。
    """)
    st.stop()

# 主交互区：问题输入
st.header("💬 医疗问答")
question = st.text_area(
    "请输入您的医疗问题（支持中英文）:",
    placeholder="例如：What are the symptoms of basal cell carcinoma? 或 皮肤癌的症状是什么？",
    height=100,
    key="question_input"
)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    submit_btn = st.button("🚀 提交问题", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ 清除历史", use_container_width=True)

if clear_btn:
    if 'history' in st.session_state:
        del st.session_state.history
    st.rerun()

# 初始化对话历史
if 'history' not in st.session_state:
    st.session_state.history = []

# 处理问题提交
if submit_btn and question:
    with st.spinner("正在检索知识库并生成答案..."):
        start_time = time.time()
        
        try:
            # 执行问答
            result = st.session_state.generator.ask(question)
            process_time = time.time() - start_time
            
            # 将问答结果存入历史
            st.session_state.history.append({
                "question": question,
                "answer": result.get('answer', '生成失败'),
                "contexts": result.get('contexts', []),
                "retrieved_count": result.get('retrieved_count', 0),
                "time": process_time
            })
            
        except Exception as e:
            st.error(f"处理过程中出错: {e}")
            st.session_state.history.append({
                "question": question,
                "answer": f"系统错误: {str(e)}",
                "contexts": [],
                "retrieved_count": 0,
                "time": 0
            })

# 显示最近的问答结果
if st.session_state.history:
    latest = st.session_state.history[-1]
    
    st.divider()
    st.subheader("📝 答案")
    
    # 答案显示区域
    with st.container(border=True):
        st.markdown(latest['answer'])
    
    # 参考资料可折叠区域
    with st.expander(f"📚 检索参考详情（共{latest['retrieved_count']}份资料，点击展开）", expanded=False):
        if latest['contexts']:
            for i, ctx in enumerate(latest['contexts']):
                # 使用列来美观地展示元数据和内容
                col_meta, col_content = st.columns([1, 4])
                with col_meta:
                    st.metric(label=f"来源 {i+1}", value=f"相似度: {ctx['similarity']:.3f}")
                with col_content:
                    st.text_area(
                        label=f"内容预览",
                        value=ctx['content_preview'],
                        height=100,
                        key=f"ctx_{i}",
                        disabled=True,
                        label_visibility="collapsed"
                    )
        else:
            st.info("本次回答未检索到参考资料。")
    
    # 性能信息
    st.caption(f"⏱️ 处理耗时: {latest['time']:.2f} 秒 | 检索资料数: {latest['retrieved_count']}")

# 历史对话记录（可选显示）
if len(st.session_state.history) > 1:
    st.divider()
    st.subheader("💾 历史对话")
    for idx, item in enumerate(reversed(st.session_state.history[:-1])):
        with st.expander(f"Q{len(st.session_state.history)-idx-1}: {item['question'][:50]}..."):
            st.markdown(f"**答案**: {item['answer'][:200]}...")
            st.caption(f"耗时: {item['time']:.2f}s | 参考资料: {item['retrieved_count']}份")

# 页脚信息
st.divider()
st.caption("""
    🏗️ **系统架构**: Milvus (向量检索) + Qwen2.5-1.5B (答案生成)  
    📖 **知识来源**: GraphRAG-Benchmark 医疗数据集  
    ⚠️ **免责声明**: 本系统生成内容仅供参考，不能替代专业医疗建议。
""")