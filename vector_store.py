import json
import os
import sys
from typing import List, Dict, Any
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

class MedicalVectorStore:
    """医疗RAG系统向量存储与检索器"""
    
    def __init__(self, 
                 collection_name: str = "medical_knowledge_base",
                 embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化向量存储
        :param collection_name: Milvus集合名称
        :param embedding_model_name: 用于编码查询的模型（需与预处理时模型一致）
        """
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"[初始化] 连接Milvus并准备集合: {collection_name}")
        print(f"[初始化] 嵌入模型维度: {self.dimension}")
    
    def connect_to_milvus(self, host: str = "localhost", port: str = "19530"):
        """连接到已启动的 Milvus Standalone 服务器"""
        try:
            # 直接连接到指定地址和端口的独立服务
            connections.connect("default", host=host, port=port)
            print(f"  ✅ 已连接到 Milvus Standalone 服务 ({host}:{port})。")
            return True
        except Exception as e:
            print(f"  ❌ 连接 Milvus 失败: {e}")
            print("  提示: 请确保已按照步骤启动 Docker 并运行了 `standalone.bat start` 命令。")
            return False
    
    def create_collection(self):
        """创建Milvus集合（数据表）"""
        # 1. 定义字段（类似数据库表的列）
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.INT64, description="原始文本块ID"),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535, description="文本块内容"),
            FieldSchema(name="text_length", dtype=DataType.INT64, description="文本长度"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension, description="文本向量"),
        ]
        
        # 2. 创建集合模式
        schema = CollectionSchema(fields, description="医疗知识库向量存储")
        
        # 3. 创建集合
        self.collection = Collection(self.collection_name, schema, consistency_level="Strong")
        print(f"  ✅ 集合 '{self.collection_name}' 创建成功。")
        
        # 4. 为向量字段创建索引（加速检索）
        index_params = {
            "metric_type": "IP",  # 内积（余弦相似度）
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}  # 聚类中心数，值越大精度越高，检索稍慢
        }
        self.collection.create_index("embedding", index_params)
        print(f"  ✅ 向量索引创建成功 (类型: IVF_FLAT, 度量: 内积)。")
    
    def insert_chunks_from_file(self, chunks_file_path: str):
        """
        从预处理好的JSON文件读取文本块和向量，并插入Milvus
        """
        print(f"[数据插入] 从文件加载块数据: {chunks_file_path}")
        
        if not os.path.exists(chunks_file_path):
            print(f"  ❌ 文件不存在: {chunks_file_path}")
            print(f"  请先运行 `python src/preprocessor.py` 生成数据。")
            return False
        
        try:
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            print(f"  ✅ 成功加载 {len(chunks_data)} 个文本块。")
            
            # 准备批量插入的数据列表
            chunk_ids = []
            chunk_texts = []
            text_lengths = []
            embeddings = []
            
            for chunk in chunks_data:
                chunk_ids.append(chunk['chunk_id'])
                chunk_texts.append(chunk['text'])
                text_lengths.append(chunk['length'])
                # 确保嵌入向量是列表格式
                if 'embedding' in chunk:
                    if isinstance(chunk['embedding'], list):
                        embeddings.append(chunk['embedding'])
                    else:
                        # 如果是numpy数组，转换为列表
                        embeddings.append(chunk['embedding'].tolist() if hasattr(chunk['embedding'], 'tolist') else list(chunk['embedding']))
                else:
                    print(f"  ⚠️  块 {chunk['chunk_id']} 缺少嵌入向量，将跳过。")
            
            # 检查数据完整性
            if len(embeddings) != len(chunks_data):
                print("  ❌ 部分块缺少嵌入向量，插入终止。")
                return False
            
            # 构建插入数据
            entities = [
                chunk_ids,
                chunk_texts,
                text_lengths,
                embeddings
            ]
            
            # 执行插入
            print(f"  正在插入 {len(chunk_ids)} 条记录到Milvus...")
            insert_result = self.collection.insert(entities)
            
            # 插入后，需要将数据从内存写入磁盘（刷新）
            self.collection.flush()
            
            print(f"  ✅ 数据插入成功！插入数量: {insert_result.insert_count}")
            print(f"  💾 数据已持久化。")
            
            # 加载集合到内存（使数据可被检索）
            self.collection.load()
            print(f"  ✅ 集合已加载到内存，准备就绪。")
            
            return True
            
        except Exception as e:
            print(f"  ❌ 插入数据时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_similar_chunks(self, query_text: str, top_k: int = 5):
        """
        检索与查询最相似的文本块
        :param query_text: 用户查询（问题）
        :param top_k: 返回最相似的结果数量
        :return: 检索到的文本块列表
        """
        if not hasattr(self, 'collection') or self.collection.is_empty:
            print("  ⚠️  集合为空或未加载，无法检索。")
            return []
        
        # 1. 将查询文本编码为向量
        print(f"  正在编码查询: \"{query_text[:50]}...\"")
        query_embedding = self.embedding_model.encode([query_text])
        
        # 2. 准备搜索参数
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 64}  # 搜索的聚类中心数，值越大精度越高，检索稍慢
        }
        
        # 3. 执行向量相似性搜索
        results = self.collection.search(
            data=query_embedding,  # 查询向量
            anns_field="embedding",  # 搜索的向量字段
            param=search_params,
            limit=top_k,  # 返回top_k个结果
            output_fields=["chunk_id", "chunk_text", "text_length"]  # 同时返回这些字段
        )
        
        # 4. 解析并格式化结果
        retrieved_chunks = []
        if results:
            for hits in results:
                for hit in hits:
                    chunk_info = {
                        'chunk_id': hit.entity.get('chunk_id'),
                        'text': hit.entity.get('chunk_text'),
                        'length': hit.entity.get('text_length'),
                        'similarity_score': hit.score,  # 相似度分数（内积值，越大越相似）
                        'distance': 1 - hit.score  # 转换为距离（余弦距离）
                    }
                    retrieved_chunks.append(chunk_info)
        
        return retrieved_chunks
    
    def display_search_results(self, query: str, results: List[Dict], top_k: int = 3):
        """美观地展示检索结果"""
        print(f"\n🔍 查询: \"{query}\"")
        print(f"📊 返回 {len(results)} 个最相关结果 (显示前{top_k}个):")
        print("-" * 80)
        
        for i, chunk in enumerate(results[:top_k]):
            print(f"\n🏆 结果 #{i+1} (相似度: {chunk['similarity_score']:.4f})")
            print(f"   文本块ID: {chunk['chunk_id']} | 长度: {chunk['length']} 字符")
            print(f"   📄 内容预览: {chunk['text'][:150]}...")
            print("-" * 60)
    
    def test_with_sample_questions(self, questions_file_path: str, num_test_questions: int = 3):
        """
        使用问题文件中的样本来测试检索系统
        """
        print(f"\n[集成测试] 使用问题文件测试检索: {questions_file_path}")
        
        if not os.path.exists(questions_file_path):
            print(f"  ❌ 问题文件不存在: {questions_file_path}")
            return
        
        try:
            with open(questions_file_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            # 确保是列表格式
            if isinstance(questions_data, list):
                test_questions = questions_data[:num_test_questions]
            else:
                print("  ❌ 问题文件格式不是列表，无法测试。")
                return
            
            print(f"  ✅ 加载 {len(test_questions)} 个测试问题。")
            
            for i, qa_pair in enumerate(test_questions):
                question = qa_pair.get('question', '')
                if question:
                    print(f"\n{'='*80}")
                    print(f"测试 #{i+1}")
                    # 执行检索
                    results = self.search_similar_chunks(question, top_k=5)
                    # 显示结果
                    self.display_search_results(question, results, top_k=2)
                    
                    # 显示参考答案（如果存在）
                    if 'answer' in qa_pair:
                        print(f"\n💡 参考答案: {qa_pair['answer'][:200]}...")
            
            print(f"\n{'='*80}")
            print("✅ 检索测试完成！")
            
        except Exception as e:
            print(f"  ❌ 测试过程中出错: {e}")

def main():
    """主函数：完整的向量存储与检索流水线"""
    print("=" * 80)
    print("医疗RAG系统 - 向量存储与检索模块")
    print("=" * 80)
    
    # 1. 定义路径 (根据你的实际项目路径调整)
    BASE_DIR = "D:/lesson/exp4/GraphRAG-Benchmark-main"
    CHUNKS_FILE = os.path.join(BASE_DIR, "processed_data", "medical_chunks.json").replace('\\', '/')
    QUESTIONS_FILE = os.path.join(BASE_DIR, "Data", "Questions", "medical_questions.json").replace('\\', '/')
    
    # 2. 初始化向量存储
    print("\n[阶段1] 初始化向量存储系统")
    vector_store = MedicalVectorStore(collection_name="medical_knowledge_v1")
    
    # 3. 连接到Milvus
    if not vector_store.connect_to_milvus():
        print("❌ 无法连接Milvus，请检查服务是否启动。")
        print("   启动命令: `milvus-server` (在独立终端中运行)")
        return
    
    # 4. 创建集合（如果不存在）
    if not utility.has_collection(vector_store.collection_name):
        vector_store.create_collection()
    else:
        print(f"\n[信息] 集合 '{vector_store.collection_name}' 已存在。")
        vector_store.collection = Collection(vector_store.collection_name)
        # 确保集合已加载
        vector_store.collection.load()
    
    # 5. 检查并插入数据
    print(f"\n[阶段2] 准备向量数据")
    # 检查集合中是否已有数据
    if vector_store.collection.is_empty:
        print("  集合为空，开始插入数据...")
        success = vector_store.insert_chunks_from_file(CHUNKS_FILE)
        if not success:
            print("❌ 数据插入失败，流程终止。")
            return
    else:
        entity_count = vector_store.collection.num_entities
        print(f"  集合中已有 {entity_count} 条数据，跳过插入。")
    
    # 6. 进行集成测试
    print(f"\n[阶段3] 检索功能集成测试")
    vector_store.test_with_sample_questions(QUESTIONS_FILE, num_test_questions=3)
    
    # 7. 交互式查询演示
    print(f"\n[阶段4] 交互式查询演示 (输入 'quit' 退出)")
    print("-" * 80)
    
    while True:
        try:
            user_query = input("\n请输入医疗问题 (或输入 'quit' 退出): ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 退出交互式查询。")
                break
            
            if not user_query:
                continue
            
            # 执行检索
            results = vector_store.search_similar_chunks(user_query, top_k=4)
            
            if results:
                vector_store.display_search_results(user_query, results, top_k=3)
            else:
                print("  未找到相关结果。")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断。")
            break
        except Exception as e:
            print(f"  检索过程中出错: {e}")

if __name__ == "__main__":
    # Windows控制台编码设置
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    main()