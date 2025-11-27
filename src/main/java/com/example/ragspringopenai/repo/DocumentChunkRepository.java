package com.example.ragspringopenai.repo;

import com.example.ragspringopenai.model.DocumentChunk;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface DocumentChunkRepository extends JpaRepository<DocumentChunk, Long> {
    List<DocumentChunk> findByDocId(String docId);
    List<DocumentChunk> findByContactNumber(String contactNumber);
    List<DocumentChunk> findByContactNumberAndDocId(String contactNumber, String docId);
}