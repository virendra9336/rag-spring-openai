package com.example.ragspringopenai.controller;

import com.example.ragspringopenai.service.RagQueryService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/rag")
public class RagQueryController {

    private final RagQueryService service;

    public RagQueryController(RagQueryService service) {
        this.service = service;
    }

    @GetMapping("/query")
    public ResponseEntity<?> query(@RequestParam String question, @RequestParam(defaultValue = "0.70") double threshold, @RequestParam(required = false) String contactNumber) throws Exception {
        String answer = service.answerQuery(question, threshold, contactNumber);
        return ResponseEntity.ok(
                Map.of(
                        "question", question,
                        "answer", answer
                )
        );
    }
}


