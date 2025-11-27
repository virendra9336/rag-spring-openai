package com.example.ragspringopenai.util;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.InputStream;

public class PdfTextExtractor {
    public static String extractText(InputStream inputStream) throws Exception {

        byte[] pdfBytes = inputStream.readAllBytes();
        try (PDDocument document = Loader.loadPDF(pdfBytes)) {
            PDFTextStripper stripper = new PDFTextStripper();
            return stripper.getText(document);
        }
    }
}