package com.example.ragspringopenai.config;


import org.springframework.ai.chat.messages.Message;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class ConversationMemoryStore {

    private final Map<String, List<Message>> memory = new ConcurrentHashMap<>();

    public List<Message> get(String key) {
        return memory.computeIfAbsent(key, k -> new ArrayList<>());
    }

    public void append(String key, Message message) {
        memory.computeIfAbsent(key, k -> new ArrayList<>()).add(message);
    }

    public void clear(String key) {
        memory.remove(key);
    }
}

