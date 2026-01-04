import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

class MedicalRAGGenerator:
    """
    医疗RAG生成器：结合检索与生成，提供最终答案。
    使用量化后的 Qwen2.5 小模型，适合在CPU或低配置GPU上运行。
    """
    
    def __init__(self, retriever=None, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        """
        初始化生成器。
        :param retriever: 可选的检索器对象。如果提供，将自动进行检索+生成。
        :param model_name: 使用的模型名称，推荐使用量化版本，如 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4'
        """
        print(f"[生成器] 初始化中，加载模型: {model_name}")
        self.retriever = retriever
        
        # 加载Tokenizer和模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            # 自动选择运行设备（优先GPU，无则CPU）
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  模型将运行在: {self.device.upper()}")
            
            # 加载模型配置
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None, # CPU模式不使用device_map
                trust_remote_code=True
            ).to(self.device)
            
            # 如果是CPU模式，确保模型在CPU上
            if self.device == "cpu":
                self.model = self.model.cpu()
                
            print("  ✅ 模型加载成功！")
            
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            print("  提示：请确保已安装 transformers：`pip install transformers accelerate`")
            raise
    
    def build_prompt(self, question: str, context_chunks: List[str]) -> str:
        """构建给模型的提示词 (Prompt)"""
        # 将检索到的文本块合并为上下文
        context = "\n---\n".join([chunk for chunk in context_chunks if chunk])
        
        # 清晰、结构化的Prompt模板
        prompt_template = (
            "你是一个专业的医疗AI助手。请严格根据以下提供的医学资料来回答用户的问题。\n\n"
            "【医学资料】\n"
            f"{context}\n\n"
            "【用户问题】\n"
            f"{question}\n\n"
            "【回答要求】\n"
            "1. 答案必须基于上述医学资料，不要编造资料中没有的信息。\n"
            "2. 如果资料不足以回答问题，请如实说明。\n"
            "3. 回答请使用中文，做到专业、准确、简洁。\n\n"
            "【开始回答】\n"
        )
        return prompt_template
    
    def generate_from_context(self, question: str, context_chunks: List[str]) -> str:
        """基于给定的上下文生成答案"""
        if not context_chunks:
            return "抱歉，未检索到相关的医学资料，无法回答此问题。"
        
        # 1. 构建Prompt
        prompt = self.build_prompt(question, context_chunks)
        
        # 2. 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 3. 生成回答（调整参数以控制生成质量和速度）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,        # 生成的最大长度
                do_sample=True,            # 启用随机性
                temperature=0.7,           # 随机性程度 (0.1~1.0)
                top_p=0.9,                 # 核采样参数
                repetition_penalty=1.1,    # 避免重复
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 4. 解码输出
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 5. 提取新生成的回答部分（移除Prompt）
        answer = full_response.replace(prompt, "").strip()
        return answer
    
    def ask(self, question: str) -> dict:
        """
        完整的RAG问答流程（如果提供了检索器）。
        返回包含答案和检索上下文的字典。
        """
        if self.retriever is None:
            return {"error": "未提供检索器，无法进行知识检索。"}
        
        print(f"\n[问答] 处理问题: {question[:60]}...")
        
        # 步骤1: 检索相关文本块
        print("  步骤1: 检索相关资料...")
        retrieved_chunks = self.retriever.search_similar_chunks(question, top_k=4)
        
        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "未找到相关的医学资料。",
                "contexts": []
            }
        
        # 提取纯文本用于生成
        context_texts = [chunk['text'] for chunk in retrieved_chunks[:3]]  # 最多用前3个
        
        # 步骤2: 基于上下文生成答案
        print("  步骤2: 生成答案...")
        answer = self.generate_from_context(question, context_texts)
        
        # 整理检索结果信息
        contexts_info = [
            {
                "chunk_id": chunk['chunk_id'],
                "content_preview": chunk['text'][:150] + "...",
                "similarity": round(chunk['similarity_score'], 4)
            }
            for chunk in retrieved_chunks[:3]
        ]
        
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts_info,
            "retrieved_count": len(retrieved_chunks)
        }

def test_generator():
    """测试生成器是否正常工作"""
    print("=" * 60)
    print("测试本地生成器")
    print("=" * 60)
    
    # 模拟的上下文（用于测试生成功能，不依赖检索器）
    test_question = "What is basal cell carcinoma?"
    test_contexts = [
        "Basal cell carcinoma (BCC) is the most common type of skin cancer.",
        "It arises from basal cells in the lower part of the epidermis.",
        "BCC usually appears on sun-exposed areas like the face, head, and neck."
    ]
    
    try:
        # 初始化生成器（不提供检索器）
        generator = MedicalRAGGenerator(retriever=None)
        
        print(f"\n测试问题: {test_question}")
        print("生成答案中...（首次运行可能较慢）")
        
        # 生成答案
        answer = generator.generate_from_context(test_question, test_contexts)
        
        print("\n✅ 生成成功！")
        print(f"\n生成的答案:\n{'-'*40}\n{answer}\n{'-'*40}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 设置UTF-8编码（Windows环境）
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # 运行测试
    success = test_generator()
    
    if success:
        print("\n【下一步】")
        print("1. 确保 Milvus 服务正在运行 (`standalone.bat start`)")
        print("2. 创建一个新的集成脚本 (如 `rag_pipeline.py`)")
        print("3. 将 VectorStore 和 Generator 结合起来")
        print("\n示例集成代码:")
        print("""
from vector_store import MedicalVectorStore
from generator import MedicalRAGGenerator

# 初始化
vector_store = MedicalVectorStore()
vector_store.connect_to_milvus()
vector_store.collection = Collection("medical_knowledge_v1")
vector_store.collection.load()

# 创建RAG生成器
generator = MedicalRAGGenerator(retriever=vector_store)

# 提问
result = generator.ask("你的问题")
print(result['answer'])
        """)