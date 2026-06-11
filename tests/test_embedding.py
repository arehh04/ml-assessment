from src.embedding import get_embedding_model


def test_get_embedding_model_returns_model():
    model = get_embedding_model()
    assert model is not None


def test_embedding_produces_vector():
    model = get_embedding_model()
    embedding = model.embed_query("What is AI governance?")
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(v, float) for v in embedding)


def test_embedding_consistent():
    model = get_embedding_model()
    emb1 = model.embed_query("test query")
    emb2 = model.embed_query("test query")
    assert emb1 == emb2


def test_embedding_different_texts_differ():
    model = get_embedding_model()
    emb1 = model.embed_query("AI governance frameworks")
    emb2 = model.embed_query("quantum computing hardware")
    assert emb1 != emb2


def test_embed_documents_batch():
    model = get_embedding_model()
    texts = ["doc one", "doc two", "doc three"]
    embeddings = model.embed_documents(texts)
    assert len(embeddings) == 3
    assert all(len(e) == 384 for e in embeddings)


def test_custom_model_name():
    model = get_embedding_model(model_name="BAAI/bge-small-en-v1.5")
    embedding = model.embed_query("test")
    assert len(embedding) == 384
