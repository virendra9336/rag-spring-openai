package com.example.ragspringopenai.service;
import com.example.ragspringopenai.model.DocumentChunk;
import com.example.ragspringopenai.repo.DocumentChunkRepository;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

@Service
@Transactional
public class RagQueryService {

    private final EmbeddingModel embeddingModel;
    private final OpenAiChatModel openAiChatModel;
    private final DocumentChunkRepository repo;
    private final ObjectMapper mapper = new ObjectMapper();

    @Autowired
    public RagQueryService(
            EmbeddingModel embeddingModel,
            OpenAiChatModel openAiChatModel,
            DocumentChunkRepository repo
    ) {
        this.embeddingModel = embeddingModel;
        this.openAiChatModel = openAiChatModel;
        this.repo = repo;
    }

    // Generate embedding using Spring AI
    public List<Double> embed(String text) {
        float[] arr = embeddingModel.embed(text); // returns float[]
        List<Double> result = new ArrayList<>(arr.length);
        for (float f : arr) result.add((double) f);
        return result;
    }

    // Cosine similarity
    private double cosineSimilarity(List<Double> a, List<Double> b) {
        double dot = 0, magA = 0, magB = 0;
        for (int i = 0; i < a.size(); i++) {
            dot += a.get(i) * b.get(i);
            magA += a.get(i) * a.get(i);
            magB += b.get(i) * b.get(i);
        }
        return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-12);
    }

    // Search by embedding
    public List<Map.Entry<DocumentChunk, Double>> search(
            String question,
            double threshold,
            String contactNumber
    ) throws Exception {
        List<Double> queryEmbedding = embed(question);

        List<DocumentChunk> allChunks =
                (contactNumber == null || contactNumber.isBlank())
                        ? repo.findAll()
                        : repo.findByContactNumber(contactNumber);

        List<Map.Entry<DocumentChunk, Double>> ranked = new ArrayList<>();

        for (DocumentChunk chunk : allChunks) {
            List<Double> other = mapper.readValue(
                    chunk.getEmbeddingJson(),
                    new TypeReference<List<Double>>() {}
            );

            double sim = cosineSimilarity(queryEmbedding, other);
            if (sim >= threshold) ranked.add(Map.entry(chunk, sim));
        }

        ranked.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return ranked;
    }

    // Answer query
    public String answerQuery(String question, double threshold, String contactNumber) throws Exception {
        var matches = search(question, threshold, contactNumber);

        if (matches.isEmpty()) return "No relevant information found.";

        StringBuilder context = new StringBuilder();
        for (var m : matches) context.append(m.getKey().getChunkText()).append("\n\n");

        String prompt = """
                You are an expert assistant.
                Use ONLY the following extracted document text to answer the question.
                
                ----------- CONTEXT -----------
                %s
                --------------------------------
                
                Question: %s
                
                Provide a factual answer.
                """.formatted(context, question);

        return openAiChatModel.call(prompt); // Spring AI OpenAiChatModel
    }
}

