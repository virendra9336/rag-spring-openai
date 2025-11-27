package com.example.ragspringopenai.service;

import com.example.ragspringopenai.model.DocumentChunk;
import com.example.ragspringopenai.repo.DocumentChunkRepository;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.ai.embedding.EmbeddingModel;

import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class EmbeddingService {

    @Autowired
    private EmbeddingModel embeddingModel;

    private final DocumentChunkRepository repo;
    private final ObjectMapper mapper = new ObjectMapper();

    @Autowired
    public EmbeddingService(DocumentChunkRepository repo) {
        this.repo = repo;
    }

    // Corrected: embed() returns List<float[]>
    public List<Double> getEmbedding(String text) {
        List<float[]> embeddings = embeddingModel.embed(List.of(text));
        float[] embArray = embeddings.get(0);
        // convert float[] → List<Double>
        List<Double> embList = new ArrayList<>();
        for (float f : embArray) embList.add((double) f);
        return embList;
    }

    public DocumentChunk saveChunk(String docId, String chunkText, List<Double> embedding) throws Exception {
        String embJson = mapper.writeValueAsString(embedding);
        DocumentChunk chunk = DocumentChunk.builder()
                .docId(docId)
                .chunkText(chunkText)
                .embeddingJson(embJson)
                .createdAt(Instant.now())
                .build();
        return repo.save(chunk);
    }

    public List<Map.Entry<DocumentChunk, Double>> searchByEmbedding(List<Double> queryEmbedding, double threshold) throws Exception {
        List<DocumentChunk> all = repo.findAll();
        List<Map.Entry<DocumentChunk, Double>> entries = new ArrayList<>();
        for (DocumentChunk dc : all) {
            List<Double> emb = mapper.readValue(dc.getEmbeddingJson(), new TypeReference<List<Double>>() {});
            if (emb.size() != queryEmbedding.size()) continue;
            double sim = cosineSimilarity(queryEmbedding, emb);
            entries.add(Map.entry(dc, sim));
        }
        entries.sort((e1, e2) -> Double.compare(e2.getValue(), e1.getValue()));
        return entries.stream().filter(e -> e.getValue() >= threshold).collect(Collectors.toList());
    }

    public static double cosineSimilarity(List<Double> a, List<Double> b) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < a.size(); i++) {
            dot += a.get(i) * b.get(i);
            na += a.get(i) * a.get(i);
            nb += b.get(i) * b.get(i);
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
    }
}
