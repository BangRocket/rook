# rook-extractors

Content extraction for multimodal memory ingestion in Rook.

## Supported Formats

- PDF documents
- DOCX documents
- Images (PNG, JPEG, GIF, WebP)
- OCR via Tesseract
- Vision LLM extraction

## Usage

```rust
use rook_extractors::pdf::PdfExtractor;

let text = PdfExtractor::extract("document.pdf")?;
```

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
