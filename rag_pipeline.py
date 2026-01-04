# rag_pipeline.py
import sys
import os

# 将项目的 src 目录添加到 Python 模块搜索路径中
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print(f"项目根目录: {project_root}")
print(f"已添加模块路径: {src_path}")

from vector_store import MedicalVectorStore
from generator import MedicalRAGGenerator
from pymilvus import Collection, utility

def main():
    print("\n" + "="*60)
    print("启动医疗RAG完整系统...")
    print("="*60)
    
    # 1. 连接向量数据库
    print("\n[1] 连接向量数据库...")
    vector_store = MedicalVectorStore(collection_name="medical_knowledge_v1")
    
    # 连接到 Milvus
    if not vector_store.connect_to_milvus():
        print("❌ 数据库连接失败，请检查 Milvus 服务是否运行 (`standalone.bat start`)")
        return
    
    # 检查集合是否存在并加载
    if not utility.has_collection(vector_store.collection_name):
        print(f"❌ 集合 '{vector_store.collection_name}' 不存在，请先运行 vector_store.py 插入数据。")
        return
    
    vector_store.collection = Collection(vector_store.collection_name)
    vector_store.collection.load()
    print(f"  ✅ 已加载集合 '{vector_store.collection_name}'")
    
    # 2. 创建RAG生成器（这可能需要一点时间加载模型）
    print("\n[2] 加载生成模型（首次使用可能需要下载）...")
    try:
        # 注意：这里将 vector_store 对象作为检索器传入
        generator = MedicalRAGGenerator(retriever=vector_store)
    except Exception as e:
        print(f"  ❌ 生成器初始化失败: {e}")
        return
    
    # 3. 交互式问答
    print("\n" + "="*60)
    print("✅ 医疗RAG系统已就绪！请输入问题（输入 'quit' 退出）")
    print("="*60)
    
    while True:
        try:
            question = input("\n🧑‍⚕️ 请输入医疗问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '退出']:
                print("👋 再见！")
                break
                
            if not question:
                continue
                
            # 执行完整的检索-生成流程
            result = generator.ask(question)
            
            print(f"\n📝 【答案】\n{'-'*40}")
            print(result['answer'])
            print(f"{'-'*40}")
            
            # 显示参考来源
            if result.get('contexts'):
                print(f"\n📚 参考了 {result.get('retrieved_count', 0)} 份资料，其中相关性最高的包括：")
                for i, ctx in enumerate(result['contexts'][:2]):  # 显示前2个
                    print(f"  {i+1}. [相关度: {ctx['similarity']:.3f}] {ctx['content_preview']}")
                    
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断。")
            break
        except Exception as e:
            print(f"\n⚠️  处理问题时出错: {e}")

if __name__ == "__main__":
    # Windows控制台编码设置
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    main()