package com.example.ragspringopenai.service;

import com.example.ragspringopenai.model.DocumentChunk;
import com.example.ragspringopenai.repo.DocumentChunkRepository;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Service
@Transactional
public class EmbeddingService {

    private final EmbeddingModel embeddingModel;
    private final OpenAiChatModel openAiChatModel;
    private final DocumentChunkRepository repo;
    private final ObjectMapper mapper = new ObjectMapper();

    /** Cache for embeddings + RAG answers */
    private final ConcurrentHashMap<String, CachedEntry> cache = new ConcurrentHashMap<>();
    private final Duration cacheTtl = Duration.ofMinutes(20);

    /** Frequently searched questions */
    private final List<String> predefinedQuestions =
            List.of("9336323244", "capgemini", "Orange");

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

    /* ---------------------------------------------------------
     * 1) FAST EMBEDDING (Uses cache)
     * ---------------------------------------------------------*/
    public List<Double> getEmbedding(String text) {

        String key = "emb:" + text.hashCode();

        CachedEntry entry = cache.get(key);
        if (entry != null && !entry.isExpired(cacheTtl)) {
            return (List<Double>) entry.getPayload().get("value");
        }

        float[] arr = embeddingModel.embed(List.of(text)).get(0);

        List<Double> result = new ArrayList<>(arr.length);
        for (float f : arr) result.add((double) f);

        cache.put(key, new CachedEntry(Map.of("value", result)));
        return result;
    }

    /* ---------------------------------------------------------
     * 2) Save Chunk
     * --------------------------------------------------------- */
    public DocumentChunk saveChunk(
            String docId,
            String username,
            String email,
            String contactNumber,
            String chunkText,
            List<Double> embedding
    ) throws Exception {

        DocumentChunk chunk = DocumentChunk.builder()
                .docId(docId)
                .username(username)
                .email(email)
                .contactNumber(contactNumber)
                .chunkText(chunkText)
                .embeddingJson(mapper.writeValueAsString(embedding))
                .createdAt(Instant.now())
                .build();

        return repo.save(chunk);
    }

    /* ---------------------------------------------------------
     * 3) Cosine Similarity
     * --------------------------------------------------------- */
    private static double cosine(List<Double> a, List<Double> b) {

        double dot = 0, na = 0, nb = 0;

        for (int i = 0; i < a.size(); i++) {
            double x = a.get(i), y = b.get(i);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }

        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }

    /* ---------------------------------------------------------
     * 4) Vector Search
     * --------------------------------------------------------- */
    public List<Map.Entry<DocumentChunk, Double>> searchByEmbedding(
            List<Double> qEmb,
            double threshold,
            String contactNumber
    ) throws Exception {

        List<DocumentChunk> chunks = (contactNumber == null || contactNumber.isBlank())
                ? repo.findAll()
                : repo.findByContactNumber(contactNumber);

        List<Map.Entry<DocumentChunk, Double>> results = new ArrayList<>(chunks.size());

        for (DocumentChunk c : chunks) {

            List<Double> emb =
                    mapper.readValue(c.getEmbeddingJson(), new TypeReference<List<Double>>() {});

            if (emb.size() != qEmb.size()) continue;

            double sim = cosine(qEmb, emb);

            if (sim >= threshold)
                results.add(Map.entry(c, sim));
        }

        // Sort descending
        results.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return results;
    }

    /* ---------------------------------------------------------
     * 5) Full RAG Search (Fast + Cached)
     * --------------------------------------------------------- */
    public ResponseEntity<?> searchDataUserRag(
            String question,
            List<Double> qEmb,
            double threshold,
            String contactNumber
    ) throws Exception {

        String key = "rag:" + question.hashCode() + ":" + contactNumber;

        // Return from cache
        CachedEntry cached = cache.get(key);
        if (cached != null && !cached.isExpired(cacheTtl)) {
            return ResponseEntity.ok(cached.getPayload());
        }

        List<Map.Entry<DocumentChunk, Double>> matches =
                searchByEmbedding(qEmb, threshold, contactNumber);

        if (matches.isEmpty()) {
            Map<String, Object> resp = Map.of(
                    "found", false,
                    "answer", "No relevant information found."
            );
            cache.put(key, new CachedEntry(resp));
            return ResponseEntity.ok(resp);
        }

        StringBuilder ctx = new StringBuilder();
        for (var m : matches) ctx.append(m.getKey().getChunkText()).append("\n\n");

        String prompt =
                "Use ONLY the following text to answer the question:\n\n" +
                        ctx +
                        "\nQuestion: " + question +
                        "\nAnswer strictly from the above text.";

        String answer = openAiChatModel.call(prompt);

        Map<String, Object> resp = Map.of(
                "found", true,
                "matches", matches.size(),
                "answer", answer
        );

        cache.put(key, new CachedEntry(resp));
        return ResponseEntity.ok(resp);
    }

    /* ---------------------------------------------------------
     * 6) Summary API (Highly Optimized)
     * --------------------------------------------------------- */
    public ResponseEntity<?> summarizeDocument(double threshold, String contactNumber) throws Exception {

        List<String> questions = List.of(
                "Summarize this entire document.",
                "List important points.",
                "Explain the document meaning."
        );

        List<String> contextBlocks = new ArrayList<>();

        for (String q : questions) {

            List<Double> emb = getEmbedding(q);

            List<Map.Entry<DocumentChunk, Double>> matches =
                    searchByEmbedding(emb, threshold, contactNumber);

            if (matches.isEmpty()) continue;

            StringBuilder ctx = new StringBuilder();
            for (var m : matches)
                ctx.append(m.getKey().getChunkText()).append("\n\n");

            contextBlocks.add(ctx.toString());
        }

        if (contextBlocks.isEmpty()) {
            return ResponseEntity.ok(Map.of(
                    "found", false,
                    "summary", "No relevant text found."
            ));
        }

        String finalContext = String.join("\n\n", contextBlocks);

        String prompt =
                """
                Summarize the following PDF content.
                Use these symbols:
                ➤ Key Point
                ⭐ Important
                🔹 Detail

                ----- PDF CONTENT -----
                """ +
                        finalContext +
                        """

                -------------------------

                Create a clean, structured summary with bullet symbols.
                """;

        String summary = openAiChatModel.call(prompt);

        return ResponseEntity.ok(Map.of(
                "found", true,
                "summary", summary
        ));
    }

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


    /* ---------------------------------------------------------
     * 7) Cached Object Wrapper
     * --------------------------------------------------------- */
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


    /* ---------------------------------------------------------
     * 8) Extract Resume Summary (Name, Email, Exp, Skills, Summary)
     * --------------------------------------------------------- */
    public ResponseEntity<?> extractResumeSummary(double threshold, String contactNumber) throws Exception {

        // 1) Questions used for semantic search
        List<String> questions = List.of(
                "Extract personal details like name, email, phone.",
                "Identify total experience.",
                "Identify primary technologies and skills.",
                "Summarize the resume in short bullets."
        );

        List<String> contextBlocks = new ArrayList<>();

        for (String q : questions) {
            List<Double> emb = getEmbedding(q);

            List<Map.Entry<DocumentChunk, Double>> matches =
                    searchByEmbedding(emb, threshold, contactNumber);

            if (matches.isEmpty()) continue;

            StringBuilder ctx = new StringBuilder();
            for (var m : matches)
                ctx.append(m.getKey().getChunkText()).append("\n\n");

            contextBlocks.add(ctx.toString());
        }

        if (contextBlocks.isEmpty()) {
            return ResponseEntity.ok(Map.of(
                    "found", false,
                    "summary", "No relevant resume information found."
            ));
        }

        String finalContext = String.join("\n\n", contextBlocks);

        // 2) Prompt for extracting structured JSON
        String prompt =
                """
                You are a Resume Extractor AI.
                Based ONLY on the following text, extract:
    
                1) Full Name  
                2) Email Address  
                3) Phone/Mobile Number  
                4) Total Experience  
                5) Technologies  
                6) Summary of the candidate in bullet points (➤, 🔹, ⭐)
    
                Return STRICT JSON only:
    
                {
                  "name": "",
                  "email": "",
                  "phone": "",
                  "experience": "",
                  "technologies": [],
                  "summary": []
                }
    
                -------- RESUME CONTENT --------
                """ +
                        finalContext +
                        """
                --------------------------------
                Produce ONLY valid JSON. Do NOT wrap inside ```json.
                """;

        String jsonOutput = openAiChatModel.call(prompt);

        // ⭐ FIX — Remove ```json or backticks so Jackson can parse
        jsonOutput = jsonOutput
                .replace("```json", "")
                .replace("```", "")
                .trim();

        // Parse JSON safely
        Map<String, Object> map = mapper.readValue(jsonOutput, new TypeReference<>() {});

        return ResponseEntity.ok(Map.of(
                "found", true,
                "resumeSummary", map
        ));
    }


}
