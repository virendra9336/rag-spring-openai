package com.example.ragspringopenai.controller;

import com.example.ragspringopenai.service.EmbeddingService;
import com.example.ragspringopenai.util.PdfTextExtractor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api")
public class DocumentController {

    private final EmbeddingService embeddingService;

    @Autowired
    public DocumentController(EmbeddingService embeddingService) {
        this.embeddingService = embeddingService;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> uploadPdf(@RequestParam("file") MultipartFile file, @RequestParam(value = "docId", required = false) String docId) throws Exception {
        if (docId == null || docId.isBlank()) docId = java.util.UUID.randomUUID().toString();
        String text = PdfTextExtractor.extractText(file.getInputStream());
        List<String> chunks = chunkText(text, 800);
        for (String c : chunks) {
            var emb = embeddingService.getEmbedding(c);
            embeddingService.saveChunk(docId, c, emb);
        }
        return ResponseEntity.ok().body("Uploaded and indexed docId=" + docId + " chunks=" + chunks.size());
    }

    @GetMapping("/search")
    public ResponseEntity<?> search(@RequestParam("q") String q, @RequestParam(value = "threshold", defaultValue = "0.78") double threshold) throws Exception {
        var qemb = embeddingService.getEmbedding(q);
        var matches = embeddingService.searchByEmbedding(qemb, threshold);
        boolean found = !matches.isEmpty();
        List<Object> response = new ArrayList<>();
        for (var e : matches) {
            response.add(java.util.Map.of("chunkId", e.getKey().getId(), "score", e.getValue(), "text", e.getKey().getChunkText()));
        }
        return ResponseEntity.ok(java.util.Map.of("found", found, "matches", response));
    }

    private List<String> chunkText(String text, int maxChars) {
        List<String> chunks = new ArrayList<>();
        if (text == null) return chunks;
        text = text.replaceAll("\\r", " ");
        int start = 0;
        while (start < text.length()) {
            int end = Math.min(start + maxChars, text.length());
            if (end < text.length()) {
                int lastSpace = text.lastIndexOf(' ', end);
                if (lastSpace > start) end = lastSpace;
            }
            chunks.add(text.substring(start, end).trim());
            start = end + 1;
        }
        return chunks;
    }
}