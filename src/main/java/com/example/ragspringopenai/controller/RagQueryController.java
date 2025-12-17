package com.example.ragspringopenai.controller;

import com.example.ragspringopenai.service.QueryService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/rag")
public class RagQueryController {

    private final QueryService service;

    public RagQueryController(QueryService service) {
        this.service = service;
    }

    @GetMapping("/query")
    public ResponseEntity<?> lanChainQuery(@RequestParam String question, @RequestParam(defaultValue = "0.70") double threshold, @RequestParam(required = false) String contactNumber) throws Exception {
        String answer = service.queryFromLangChain(question, threshold, contactNumber);
        return ResponseEntity.ok(
                Map.of(
                        "question", question,
                        "answer", answer
                )
        );
    }
}


