from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType
from configuration.config import *

class IndexUtil:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
        )
        # 嵌入模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=r'D:\Work\tools\models\bge-base-zh-v1.5',
            encode_kwargs={'normalize_embeddings': True},
        )

    # 创建全文索引，传入索引名称，节点标签，属性
    def create_fulltext_index(self, index_name, label, property):
        cypher = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS 
            FOR (n:{label}) ON EACH [n.{property}]
        """
        self.graph.query(cypher)

    # 创建向量索引，需要传入生成向量的 “源属性”，以及嵌入向量的 “目标属性”
    '''
    这段代码的作用是先把图谱节点上的文本属性转成 embedding 向量，写回 Neo4j 节点属性中，
    然后基于该向量属性创建 Neo4j 的 vector index。这样图谱就具备了语义检索能力，后续可以支持相似节点召回、
    向量搜索以及图谱问答场景'''
    def create_vector_index(self, index_name, label, source_property, embedding_property):
        # 生成嵌入向量，并添加到节点属性中
        embedding_dim = self._add_embedding(label, source_property, embedding_property)

        cypher = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON n.{embedding_property}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """
        self.graph.query(cypher)

    # 内部函数：生成嵌入向量，并添加到节点属性中
    def _add_embedding(self, label,source_property, embedding_property):
        # 1. 查询所有节点对应的源属性值，作为模型的输入；还需要查出节点id
        cypher = f"""
            MATCH (n:{label})
            RETURN n.{source_property} AS text, id(n) AS id
        """
        results = self.graph.query(cypher)

        # 2. 获取查询结果中的文本内容
        docs = [result['text'] for result in results]

        # 3. 调用嵌入模型，得到嵌入向量
        embeddings = self.embedding_model.embed_documents(docs)

        # 4. 将id和嵌入向量组合成字典形式
        batch = []
        for result, embedding in zip(results, embeddings):
            item = {'id': result['id'], 'embedding': embedding}
            batch.append(item)

        # 5. 执行cypher，按id查节点，写入新的嵌入向量属性
        cypher = f"""
            UNWIND $batch AS item
            MATCH (n:{label})
            WHERE id(n) = item.id
            SET n.{embedding_property} = item.embedding
        """
        self.graph.query(cypher, params={"batch": batch})

        return len(embeddings[0])

if __name__ == "__main__":
    index = IndexUtil()
    index.create_fulltext_index(index_name="trademark_fulltext_index", label="Trademark", property="name")
    index.create_vector_index(index_name="trademark_vector_index", label="Trademark", source_property="name",
                              embedding_property="embedding")

    # index_name = "trademark_vector_index"  # default index name
    # keyword_index_name = "trademark_fulltext_index"  # default keyword index name
    #
    # store = Neo4jVector.from_existing_index(
    #     index.embedding_model,
    #     url=NEO4J_CONFIG["uri"],
    #     username=NEO4J_CONFIG["auth"][0],
    #     password=NEO4J_CONFIG["auth"][1],
    #     index_name=index_name,
    #     keyword_index_name=keyword_index_name,
    #     search_type=SearchType.HYBRID
    # )
    #
    # result=store.similarity_search("Apple",k=5)
    #
    # print(result)

    index.create_fulltext_index(index_name='spu_fulltext_index', label='SPU', property='name')
    index.create_vector_index(index_name='spu_vector_index', label='SPU', source_property='name',embedding_property='embedding')
    index.create_fulltext_index(index_name='sku_fulltext_index', label='SKU', property='name')
    index.create_vector_index(index_name='sku_vector_index', label='SKU', source_property='name', embedding_property='embedding')

    index.create_fulltext_index(index_name='category1_fulltext_index', label='Category1', property='name')
    index.create_vector_index(index_name='category1_vector_index', label='Category1', source_property='name',embedding_property='embedding')
    index.create_fulltext_index(index_name='category2_fulltext_index', label='Category2', property='name')
    index.create_vector_index(index_name='category2_vector_index', label='Category2', source_property='name',embedding_property='embedding')
    index.create_fulltext_index(index_name='category3_fulltext_index', label='Category3', property='name')
    index.create_vector_index(index_name='category3_vector_index', label='Category3', source_property='name',embedding_property='embedding')