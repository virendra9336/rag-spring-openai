package com.example.ragspringopenai.service;
import com.example.ragspringopenai.config.ConversationMemoryStore;
import com.example.ragspringopenai.model.DocumentChunk;
import com.example.ragspringopenai.repo.DocumentChunkRepository;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.AssistantMessage;

import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.embedding.EmbeddingModel;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


@Service
@RequiredArgsConstructor
public class QueryService {

    private final DocumentChunkRepository repo;
    private final OpenAiChatModel chatModel;
    private final EmbeddingModel embeddingModel;
    private final ConversationMemoryStore memoryStore;
    private final ObjectMapper mapper = new ObjectMapper();

    public String queryFromLangChain(String question, double threshold, String contactNumber) throws Exception {

        var matches = search(question, threshold, contactNumber);
        if (matches.isEmpty()) return "No relevant information found.";

        StringBuilder context = new StringBuilder();
        for (var m : matches) {
            context.append(m.getKey().getChunkText()).append("\n\n");
        }

        List<Message> messages = new ArrayList<>(memoryStore.get(contactNumber));

        messages.add(new SystemMessage("""
        You are an expert assistant.
        Use ONLY the provided context.
        
        -------- CONTEXT --------
        %s
        -------------------------
        """.formatted(context)));

        UserMessage userMessage = new UserMessage(question);
        messages.add(userMessage);

        ChatResponse response = chatModel.call(new Prompt(messages));
        String answer = response.getResult().getOutput().getText();
        memoryStore.append(contactNumber, userMessage);
        memoryStore.append(contactNumber, new AssistantMessage(answer));
        return answer;
    }


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
                    new TypeReference<>() {}
            );

            double sim = cosineSimilarity(queryEmbedding, other);
            if (sim >= threshold) ranked.add(Map.entry(chunk, sim));
        }

        ranked.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return ranked;
    }

    private List<Double> embed(String text) {
        float[] arr = embeddingModel.embed(text);
        List<Double> result = new ArrayList<>(arr.length);
        for (float f : arr) result.add((double) f);
        return result;
    }

    private double cosineSimilarity(List<Double> a, List<Double> b) {
        double dot = 0, magA = 0, magB = 0;
        for (int i = 0; i < a.size(); i++) {
            dot += a.get(i) * b.get(i);
            magA += a.get(i) * a.get(i);
            magB += b.get(i) * b.get(i);
        }
        return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-12);
    }
}

