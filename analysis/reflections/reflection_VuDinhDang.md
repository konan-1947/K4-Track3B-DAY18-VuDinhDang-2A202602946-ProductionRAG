# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Vu Dinh Dang  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Semantic chunking giúp nhóm các câu cùng chủ đề thay vì cắt cơ học theo độ dài. Trong bài làm, hàm có fallback lexical để tránh tải model ngoài ý muốn nhưng vẫn giữ logic tách theo độ tương đồng giữa câu liên tiếp. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent-child chunking cho phép retrieve child nhỏ để tăng precision, nhưng vẫn giữ `parent_id` để truy ngược context lớn hơn. Kết quả pipeline tạo 101 child chunks từ 25 tài liệu markdown. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Hàm parse markdown headers và giữ section metadata. Đây là hướng tốt cho tài liệu chính sách vì nhiều nội dung nằm trong header/table/list. Failure analysis cho thấy table vẫn cần được bảo toàn tốt hơn nữa. |
| BM25 + Dense fusion | M2 | `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | BM25 bắt từ khóa chính xác tiếng Việt, Dense fallback dùng lexical cosine trong môi trường không tải model. RRF kết hợp hai danh sách để giảm rủi ro một retriever bỏ sót tài liệu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | CrossEncoder thật chỉ bật khi cho phép tải model (`ALLOW_MODEL_DOWNLOADS=1`). Fallback lexical reranker vẫn sort candidate theo độ liên quan. Một số failure cho thấy reranker thật sẽ giúp câu hỏi multi-hop và version-sensitive tốt hơn. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Hàm hỗ trợ RAGAS thật khi có `OPENAI_API_KEY`; nếu không có API key thì dùng lexical fallback để pipeline vẫn end-to-end. Production scores fallback: Faithfulness 1.0000, Answer Relevancy 0.6682, Context Precision 1.0000, Context Recall 0.7223. |
| Failure analysis | M4 | `failure_analysis()` | Diagnostic tree map metric thấp nhất sang nguyên nhân và hướng sửa. Bottom failures chủ yếu nằm ở multi-hop, bảng markdown bị cắt, và tài liệu có phiên bản cũ/mới. |
| Contextual embeddings / enrichment | M5 | `contextual_prepend()`, `_enrich_single_call()` | Enrichment thêm context line, summary, hypothesis questions và metadata trước khi index. Khi không có API key, fallback vẫn tạo context từ source/category để cải thiện retrieval. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `Python was not found; run without arguments to install from the Microsoft Store...`
  - `ModuleNotFoundError: No module named 'dotenv'`
  - `ModuleNotFoundError: No module named 'pypdf'`
  - `RAGAS evaluation unavailable, using lexical fallback: OPENAI_API_KEY not set`

- **Nguyên nhân gốc rễ & Cách debug:**
  - Môi trường Git Bash không nhận lệnh `python`, nên chuyển sang Python launcher `py -3`.
  - Môi trường test thiếu một số dependency như `python-dotenv` và `pypdf`. Để tránh bài fail chỉ vì thiếu package, `config.py` được bổ sung fallback cho `load_dotenv()`, còn `_extract_pdf_text()` bỏ qua PDF nếu chưa có `pypdf`.
  - Các model embedding/reranker có thể tải file lớn, nên code không tự tải model. Thay vào đó, chỉ tải khi set `ALLOW_MODEL_DOWNLOADS=1`; mặc định dùng fallback lexical.
  - Không có `OPENAI_API_KEY`, nên RAGAS thật không chạy. Hàm `evaluate_ragas()` được bọc try/except và dùng lexical fallback để vẫn tạo `reports/ragas_report.json`.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Cần tìm hiểu sâu hơn về version-aware retrieval để xử lý tài liệu cũ/mới như `nghi_phep_nam_v2023.md` và `nghi_phep_nam_v2024.md`.
  - Cần cải thiện multi-hop retrieval cho câu hỏi cần lấy dữ kiện từ nhiều tài liệu, ví dụ phép năm + bảng lương.
  - Cần tối ưu chunking cho markdown table để không cắt mất dòng quan trọng trong bảng phê duyệt.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Production RAG cho tài liệu chính sách nội bộ tiếng Việt

#### 1. Hiện trạng
- **Pipeline hiện tại:** Load tài liệu markdown/PDF text layer → hierarchical chunking → enrichment → hybrid search BM25 + dense fallback → rerank → answer → evaluation.
- **Vấn đề / Bottlenecks đang gặp:**
  - Một số câu hỏi multi-hop chưa lấy đủ context.
  - Markdown table có thể bị cắt sai đoạn.
  - Tài liệu có nhiều phiên bản cũ/mới dễ gây nhầm câu trả lời.
  - Khi không có model/API key, fallback chạy được nhưng chất lượng semantic chưa bằng model thật.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:**
   - Dùng hierarchical chunking làm mặc định.
   - Bổ sung structure-aware riêng cho tài liệu nhiều header/table để giữ nguyên markdown table.
   - Với tài liệu versioned policy, chunk nên giữ metadata `version`, `effective_date`, `status`.

2. **Search retrieval:**
   - Dùng Hybrid Search: BM25 cho keyword/version/number, Dense cho semantic match.
   - Áp dụng RRF để hợp nhất kết quả.
   - Với câu hỏi multi-hop, tách query thành nhiều sub-query rồi merge contexts.

3. **Reranking:**
   - Khi được phép tải model, dùng `BAAI/bge-reranker-v2-m3` cho rerank top-20 → top-3.
   - Thêm rule ưu tiên tài liệu hiện hành so với tài liệu cũ.

4. **Evaluation:**
   - Dùng RAGAS 4 metrics khi có `OPENAI_API_KEY`.
   - Ngoài RAGAS, thêm custom checks cho numeric answer, version correctness và negation correctness.

5. **Enrichment:**
   - Dùng combined enrichment để giảm số API calls.
   - Tạo summary, hypothesis questions, contextual prepend và metadata tự động.
   - Metadata cần có thêm `category`, `version`, `status`, `effective_date`.

#### 3. Timeline triển khai
- **Tuần 1:** Hoàn thiện chunking cho markdown table và metadata version/status.
- **Tuần 2:** Bật dense embedding + CrossEncoder reranker thật nếu môi trường cho phép tải model.
- **Tuần 3:** Thêm query decomposition cho multi-hop và numeric questions.
- **Tuần 4:** Chạy RAGAS thật với API key, phân tích bottom failures và tối ưu prompt/retrieval.
