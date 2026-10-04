# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Vu Dinh Dang  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

Do môi trường chạy không có `OPENAI_API_KEY`, báo cáo được sinh bằng lexical fallback trong `evaluate_ragas()` thay vì RAGAS API thật. Pipeline vẫn chạy end-to-end và tạo `reports/ragas_report.json`.

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | N/A | 1.0000 | N/A |
| Answer Relevancy | N/A | 0.6682 | N/A |
| Context Precision | N/A | 1.0000 | N/A |
| Context Recall | N/A | 0.7223 | N/A |

## Bottom-5 Failures

### #1
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu cần Giám đốc phòng ban phê duyệt, cần xác nhận cấu hình kỹ thuật từ CNTT và ít nhất 3 báo giá.
- **Got:** Context/answer ưu tiên nhầm sang `dao_tao_noi_bo.md` và chỉ lấy được một phần từ `mua_sam.md`.
- **Worst metric:** answer_relevancy = 0.4000
- **Error Tree:** Output sai → Context chưa đủ → Query multi-hop giữa giá trị mua sắm + laptop/CNTT + báo giá → Retrieval chưa gom đủ section liên quan.
- **Root cause:** Chunk mua sắm bị chia nhỏ, bảng phê duyệt và lưu ý CNTT nằm ở các child chunks khác nhau; fallback reranker lexical chưa đủ mạnh để ưu tiên đầy đủ các chunk.
- **Suggested fix:** Tăng top_k trước rerank, giữ bảng markdown nguyên vẹn bằng structure-aware chunking, thêm metadata category `procurement/it`, dùng CrossEncoder thật khi được phép tải model.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép; lương Senior P3-P4 là 20-35 triệu VNĐ/tháng.
- **Got:** Trả lời đúng phần phép năm nhưng thiếu phần lương Senior.
- **Worst metric:** context_recall = 0.4783
- **Error Tree:** Output thiếu → Context chỉ có nghỉ phép → Query multi-hop cần cả chính sách phép năm và bảng lương → Retrieval không lấy chunk bảng lương.
- **Root cause:** Câu hỏi yêu cầu join 2 tài liệu (`nghi_phep_nam_v2024.md` + `bang_luong_2024.md`), nhưng top contexts bị chiếm bởi các chunk nghỉ phép.
- **Suggested fix:** Dùng query decomposition thành 2 sub-query: “9 năm thâm niên nghỉ phép” và “lương Senior P3-P4”; sau đó merge contexts bằng RRF.

### #3
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất 20.000.000 VNĐ/tháng; lương thử việc = 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Trả về quy định 85% lương thử việc nhưng thiếu bảng lương Junior để tính ra con số 17 triệu.
- **Worst metric:** context_recall = 0.4286
- **Error Tree:** Output thiếu phép tính → Context thiếu bảng lương → Query cần multi-hop giữa thử việc và bảng lương → Retrieval chỉ lấy `thu_viec.md`.
- **Root cause:** Pipeline chưa có bước reasoning/calculation và chưa lấy được chunk `bang_luong_2024.md`.
- **Suggested fix:** Thêm retrieval theo entity “Junior”, tăng hybrid top_k, và bổ sung bước answer synthesis tính toán số học từ context.

### #4
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Lấy đúng tài liệu mua sắm nhưng chunk bị cắt ở đầu bảng, chưa chứa dòng “trên 50 triệu → CEO”.
- **Worst metric:** answer_relevancy = 0.4615
- **Error Tree:** Output chưa đủ → Context đúng file nhưng sai đoạn trong bảng → Chunking cắt bảng markdown → Rerank chọn chunk đầu bảng thay vì dòng cần thiết.
- **Root cause:** Hierarchical child chunk size nhỏ làm bảng phê duyệt bị tách nhiều đoạn.
- **Suggested fix:** Structure-aware chunking cho bảng, không cắt giữa table; hoặc tăng child_size cho tài liệu có markdown table.

### #5
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Chính sách hiện hành v2024 là 15 ngày phép năm; v2023 12 ngày đã bị thay thế.
- **Got:** Retrieval ưu tiên nhầm nghỉ phép đặc biệt và chính sách v2023; thiếu chunk v2024 trong top contexts.
- **Worst metric:** context_recall = 0.4800
- **Error Tree:** Output sai phiên bản → Context thiếu bản hiện hành → Query version-sensitive → Retrieval chưa ưu tiên metadata version/current.
- **Root cause:** Corpus có tài liệu cũ và mới; fallback lexical search không phân biệt trạng thái “hiện hành/đã thay thế”.
- **Suggested fix:** Extract metadata `version`, `effective_date`, `status`; ưu tiên tài liệu hiện hành trong reranking hoặc filter theo version.

## Case Study (cho presentation)

**Question chọn phân tích:** Nhân viên được nghỉ bao nhiêu ngày phép năm?

**Error Tree walkthrough:**
1. Output đúng? → Chưa đúng vì câu trả lời bị nhiễu bởi nghỉ phép đặc biệt/v2023.
2. Context đúng? → Chưa đủ; context thiếu hoặc xếp thấp chunk v2024 có “15 ngày phép năm”.
3. Query rewrite OK? → Chưa; query cần nhấn mạnh “phép năm hiện hành”.
4. Fix ở bước: Retrieval + reranking + metadata versioning.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm metadata extraction cho `version`, `effective_date`, `status`.
- Áp dụng structure-aware chunking riêng cho markdown tables.
- Tăng candidate retrieval top_k trước rerank.
- Nếu được phép tải model, bật `ALLOW_MODEL_DOWNLOADS=1` để dùng `BAAI/bge-m3` và `BAAI/bge-reranker-v2-m3` thật.
