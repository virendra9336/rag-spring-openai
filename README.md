# RAG with Spring AI (OpenAI) + Postgres (docker-compose)

This sample project demonstrates:
- Upload a PDF and extract text (Apache PDFBox)
- Chunk text, compute embeddings via **Spring AI** OpenAI starter
- Store chunks + embeddings in Postgres (embedding stored as JSON)
- Search by computing query embedding and doing cosine-similarity in Java (simple RAG)

## How to run
1. Set your OpenAI API key as environment variable:
   `export OPENAI_API_KEY="sk-..."`

2. Start Postgres:
   `docker-compose up -d`

3. Build and run the app:
   `./mvnw spring-boot:run` or `mvn spring-boot:run`

4. Upload PDF:
   `curl -F file=@sample.pdf http://localhost:8080/api/upload`

5. Search:
   `curl "http://localhost:8080/api/search?q=your+search+text"`

Notes:
- This uses Spring AI `org.springframework.ai:spring-ai-starter-model-openai:1.1.0`. You may change model names in the code (embeddings model).
- For production, consider vector DBs (pgvector, Milvus, Pinecone) or use pgvector extension for more efficient similarity search.
