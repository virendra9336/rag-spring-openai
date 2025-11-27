package com.example.ragspringopenai.model;

import jakarta.persistence.*;
import lombok.*;
import java.time.Instant;

@Entity
@Table(name = "document_chunk")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DocumentChunk {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String docId;

    @Lob
    @Column(columnDefinition = "text")
    private String chunkText;

    @Lob
    @Column(columnDefinition = "text")
    private String embeddingJson;

    private Instant createdAt;
}
