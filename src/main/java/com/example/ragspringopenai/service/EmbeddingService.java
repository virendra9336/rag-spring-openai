
package com.example.ragspringopenai.service;
import com.example.ragspringopenai.model.DocumentChunk;
import com.example.ragspringopenai.repo.DocumentChunkRepository;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Service
@Transactional
public class EmbeddingService {

    private final EmbeddingModel embeddingModel;
    private final OpenAiChatModel openAiChatModel;
    private final DocumentChunkRepository repo;
    private final ObjectMapper mapper = new ObjectMapper();

    // Simple in-memory cache: key -> CachedEntry
    private final ConcurrentHashMap<String, CachedEntry> cache = new ConcurrentHashMap<>();
    // TTL for cached answers
    private final Duration cacheTtl = Duration.ofMinutes(10);

    @Autowired
    public EmbeddingService(
            EmbeddingModel embeddingModel,
            OpenAiChatModel openAiChatModel,
            DocumentChunkRepository repo
    ) {
        this.embeddingModel = embeddingModel;
        this.openAiChatModel = openAiChatModel;
        this.repo = repo;
    }

    /* ---------------------------------------------------------------------
     * 1. Generate Embedding
     * --------------------------------------------------------------------- */

    public List<Double> getEmbedding(String text) {
        List<float[]> embeddings = embeddingModel.embed(List.of(text));
        float[] arr = embeddings.get(0);

        List<Double> list = new ArrayList<>(arr.length);
        for (float f : arr) {
            list.add((double) f);
        }
        return list;
    }

    /* ---------------------------------------------------------------------
     * 2. Save Chunk
     * --------------------------------------------------------------------- */
    public DocumentChunk saveChunk(
            String docId,
            String username,
            String email,
            String contactNumber,
            String chunkText,
            List<Double> embedding
    ) throws Exception {

        String embJson = mapper.writeValueAsString(embedding);

        DocumentChunk chunk = DocumentChunk.builder()
                .docId(docId)
                .username(username)
                .email(email)
                .contactNumber(contactNumber)
                .chunkText(chunkText)
                .embeddingJson(embJson)
                .createdAt(Instant.now())
                .build();

        return repo.save(chunk);
    }

    /* ---------------------------------------------------------------------
     * 3. Generic Search (Overloaded)
     * --------------------------------------------------------------------- */
    public List<Map.Entry<DocumentChunk, Double>> searchByEmbedding(
            List<Double> queryEmbedding,
            double threshold
    ) throws Exception {
        return searchByEmbedding(queryEmbedding, threshold, null);
    }

    /* ---------------------------------------------------------------------
     * 4. Search by Embedding (Vector Search)
     * --------------------------------------------------------------------- */
    public List<Map.Entry<DocumentChunk, Double>> searchByEmbedding(
            List<Double> queryEmbedding,
            double threshold,
            String contactNumber
    ) throws Exception {

        List<DocumentChunk> chunks =
                (contactNumber == null || contactNumber.isBlank())
                        ? repo.findAll()
                        : repo.findByContactNumber(contactNumber);

        List<Map.Entry<DocumentChunk, Double>> results = new ArrayList<>();

        for (DocumentChunk chunk : chunks) {

            List<Double> chunkEmbedding =
                    mapper.readValue(chunk.getEmbeddingJson(), new TypeReference<List<Double>>() {});

            // Skip corrupted embeddings
            if (chunkEmbedding.size() != queryEmbedding.size()) {
                continue;
            }

            double score = cosineSimilarity(queryEmbedding, chunkEmbedding);
            results.add(Map.entry(chunk, score));
        }

        return results.stream()
                .filter(e -> e.getValue() >= threshold)
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue())) // descending
                .collect(Collectors.toList());
    }

    /* ---------------------------------------------------------------------
     * 5. Full RAG Search – Retrieve → Enhance → Generate
     *    Uses simple in-memory cache for identical questions.
     * --------------------------------------------------------------------- */
    public ResponseEntity<?> searchDataUserRag(String question, List<Double> queryEmbedding, double threshold, String contactNumber) throws Exception {

        String key = cacheKey(question, contactNumber);
        CachedEntry cached = cache.get(key);
        if (cached != null && !cached.isExpired(cacheTtl)) {
            return ResponseEntity.ok(cached.getPayload());
        }

        List<Map.Entry<DocumentChunk, Double>> matches =
                searchByEmbedding(queryEmbedding, threshold, contactNumber);

        if (matches.isEmpty()) {
            Map<String, Object> resp = Map.of(
                    "found", false,
                    "answer", "No relevant information found in the PDF."
            );
            cache.put(key, new CachedEntry(resp));
            return ResponseEntity.ok(resp);
        }

        StringBuilder context = new StringBuilder();
        matches.forEach(m -> context.append(m.getKey().getChunkText()).append("\n\n"));

        String prompt =
                "Use ONLY the following text to answer the question:\n\n"
                        + context
                        + "\n\nQuestion: " + question
                        + "\nAnswer from the text only.";

        // FIX → Using Spring AI OpenAIChatModel
        String answer = openAiChatModel.call(prompt);

        Map<String, Object> resp = Map.of(
                "found", true,
                "matches", matches.size(),
                "answer", answer
        );

        cache.put(key, new CachedEntry(resp));
        return ResponseEntity.ok(resp);
    }

    private String cacheKey(String question, String contactNumber) {
        return (contactNumber == null ? "" : contactNumber.trim()) + "::" + (question == null ? "" : question.trim());
    }

    /* ---------------------------------------------------------------------
     * Utility: Cosine Similarity
     * --------------------------------------------------------------------- */
    public static double cosineSimilarity(List<Double> a, List<Double> b) {
        double dot = 0.0, na = 0.0, nb = 0.0;

        for (int i = 0; i < a.size(); i++) {
            dot += a.get(i) * b.get(i);
            na += a.get(i) * a.get(i);
            nb += b.get(i) * b.get(i);
        }

        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }
    /* ---------------------------------------------------------------------
     * Utility: Chunk Text
     * --------------------------------------------------------------------- */
    public List<String> chunkText(String text, int maxChars) {
        List<String> chunks = new ArrayList<>();
        if (text == null) return chunks;
        text = text.replaceAll("\\r", " ").trim();
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(start + maxChars, text.length());
            if (end < text.length()) {
                int lastSpace = text.lastIndexOf(' ', end);
                if (lastSpace > start) end = lastSpace;
            }
            String chunk = text.substring(start, end).trim();
            if (!chunk.isEmpty()) chunks.add(chunk);
            start = end + 1;
        }
        return chunks;
    }

    // Simple cache entry wrapper
    private static class CachedEntry {
        private final Map<String, Object> payload;
        private final Instant createdAt;

        CachedEntry(Map<String, Object> payload) {
            this.payload = payload;
            this.createdAt = Instant.now();
        }

        boolean isExpired(Duration ttl) {
            return Instant.now().isAfter(createdAt.plus(ttl));
        }

        Map<String, Object> getPayload() {
            return payload;
        }
    }
}
