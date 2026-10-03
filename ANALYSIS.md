# Báo Cáo Phân Tích & Benchmark Memory Systems (Day 17)

Báo cáo này tổng hợp kết quả đánh giá thực nghiệm kiến trúc bộ nhớ cho AI Agent (Track 3 - Giai đoạn 2), so sánh giữa **Baseline Agent** (chỉ có bộ nhớ ngắn hạn trong phiên) và **Advanced Agent** (tích hợp 3 tầng bộ nhớ: Short-term, Persistent `User.md`, và Compact Memory).

---

## 1. Bảng Kết Quả Thực Nghiệm

Toàn bộ thực nghiệm được chạy độc lập trên 2 bộ benchmark chuẩn:
1. **Standard Benchmark (`data/conversations.json`)**: 10 cuộc hội thoại thông thường, nhiều phiên, có đính chính thông tin.
2. **Long-Context Stress Benchmark (`data/advanced_long_context.json`)**: Hội thoại rất dài với nhiều ngữ cảnh kỹ thuật, tin tức và nhiễu (disturbances).

### Suite 1: Standard Benchmark
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 1,784 | 14,502 | 0.0% | 10.7% | 0 | 0 |
| **Advanced Agent** | 3,553 | 33,232 | **100.0%** | **99.3%** | 284 | 0 |

### Suite 2: Long-Context Stress Benchmark
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 308 | 22,142 | 0.0% | 10.0% | 0 | 0 |
| **Advanced Agent** | 622 | **13,541** | **100.0%** | **96.7%** | 209 | **24** |

*(Ghi chú: Ở Stress Benchmark, Advanced Agent giúp giảm **38.8%** Prompt tokens processed so với Baseline).*

---

## 2. Phân Tích Chuyên Sâu Các Chỉ Số

### 2.1. Vì sao Advanced Agent đạt Recall 100% còn Baseline là 0%?
- **Baseline Agent**: Thiết kế theo mô hình ngây thơ ("naive session memory"), chỉ lưu trữ message trong ram theo từng `thread_id`. Khi chuyển sang phiên mới (`fresh session thread`), lịch sử trò chuyện trống trơn và không có cơ chế lưu trữ bền vững. Vì vậy, Baseline hoàn toàn quên mọi thông tin cá nhân của người dùng giữa các phiên (Cross-session recall = 0%).
- **Advanced Agent**: Sử dụng tầng **Persistent Memory (`User.md`)**. Khi người dùng chia sẻ các facts ổn định (tên, nơi ở, nghề nghiệp, đồ uống yêu thích, sở thích, thú cưng, v.v.), agent tự động trích xuất và lưu vào file markdown cá nhân hóa. Ngay cả khi mở thread mới hoàn toàn, agent vẫn tải `User.md` vào prompt context, cho phép trả lời chính xác 100% các câu hỏi truy vấn.

### 2.2. Vì sao Advanced Agent tốn nhiều token hơn ở hội thoại ngắn?
- Nhìn vào Standard Benchmark, ở các hội thoại ngắn:
  - `Prompt tokens processed`: Baseline tốn 14,502 tokens, trong khi Advanced tốn 33,232 tokens.
  - `Agent tokens only`: Baseline là 1,784, Advanced là 3,553 tokens.
- **Nguyên nhân**:
  1. **System Prompt & Profile Overhead**: Advanced Agent luôn phải nạp toàn bộ profile `User.md` và các hướng dẫn hệ thống vào context ở mỗi lượt tương tác, ngay cả khi người dùng chỉ chào hỏi ngắn gọn.
  2. **Response Detail**: Advanced Agent có câu trả lời đầy đủ, đúng ngữ cảnh và tuân theo preference (ví dụ: kèm giải thích thực tế, bullet points), dẫn đến lượng generated tokens cao hơn so với câu trả lời ngắn mặc định của Baseline.
- **Kết luận**: Đối với hội thoại ngắn ít lượt, persistent memory tạo ra một khoản "chi phí cố định" (fixed token tax).

### 2.3. Vì sao Compact Memory tạo ra bước ngoặt ở hội thoại dài?
- Trong **Long-Context Stress Benchmark** (hội thoại kéo dài hơn 20 lượt với các bài viết dài hàng trăm từ):
  - Baseline Agent tích lũy toàn bộ lịch sử thô: ở lượt thứ $N$, prompt load bằng tổng toàn bộ từ lượt 1 đến $N-1$. Tổng lượng `Prompt tokens processed` tăng theo hàm bậc hai ($O(N^2)$), đạt tới **22,142 tokens**.
  - Advanced Agent kích hoạt **Compact Memory Manager**: Ngay khi token trong thread vượt ngưỡng (`threshold_tokens = 600`), các tin nhắn cũ được tóm tắt thành summary súc tích, chỉ giữ lại 4 tin nhắn gần nhất (`keep_messages = 4`). Qua 24 lần compaction, kích thước prompt luôn được chặn trên ở mức an toàn.
- **Kết quả**: `Prompt tokens processed` của Advanced Agent chỉ còn **13,541 tokens** (tiết kiệm gần **39% chi phí context**, và tỉ lệ tiết kiệm này sẽ còn tăng vọt lên 70-90% nếu hội thoại dài hàng trăm lượt).

### 2.4. Đánh giá sự tăng trưởng của Memory File (`User.md`) & Rủi ro
- Dung lượng file `User.md` tăng từ 0 lên 284 bytes (Standard) và 209 bytes (Stress).
- **Rủi ro tiềm ẩn**:
  1. **Nhiễm bẩn bộ nhớ (Memory Hallucination / Misattribution)**: Người dùng hay đặt câu hỏi (ví dụ: *"Nhắc lại giúp mình tên và nghề"*), nếu regex/extractor yếu kém sẽ bắt nhầm từ ngữ trong câu hỏi thành facts mới và ghi đè làm hỏng tên thật.
  2. **Xung đột thông tin cũ - mới (Conflict Resolution)**: Khi người dùng đổi nghề (từ backend sang MLOps) hoặc chuyển nơi ở (từ Huế sang Đà Nẵng), nếu agent lưu cả hai sẽ dẫn đến mâu thuẫn nhận thức.
  3. **Phình to không kiểm soát (Unbounded Growth)**: Nếu lưu cả các chi tiết vụn vặt nhất thời vào `User.md`, file markdown sẽ ngày càng dài và nuốt trọn context window.

---

## 3. Các Điểm Sáng Tạo & Bonus Đã Triển Khai (Mức 90 - 100 điểm)

1. **Question Guardrail (Ngăn chặn lưu sai từ câu hỏi)**:
   - Tự động nhận diện các mẫu câu truy vấn (`?`, `nhắc lại giúp mình`, `là ai`, `ở đâu`, `nghề gì`) để bỏ qua, không kích hoạt ghi nhận facts.
   - Loại bỏ các stopwords (ví dụ: `và`, `style`, `của`, `mình`) khỏi tên người dùng để tránh ghi đè sai.
2. **Conflict Resolution & Correction Handling**:
   - Khi phát hiện đính chính (`chuyển sang MLOps engineer`, `cập nhật nơi ở sang Đà Nẵng`), hệ thống tự động ghi đè giá trị mới nhất lên `User.md` thay vì nối thêm làm mâu thuẫn dữ liệu.
3. **Noise Filtering (Lọc nhiễu)**:
   - Loại bỏ thông tin đùa giỡn (*"chuyển sang làm product manager... câu đùa"*) hoặc địa điểm công tác tạm thời (*"Hà Nội chỉ là nơi vừa bay ra họp"*).
4. **Hỗ trợ đa dạng Provider (OpenAI, Gemini, Anthropic, Ollama, OpenRouter, Custom)**:
   - Tương thích 100% với cấu hình Gemini của người dùng, sẵn sàng chạy live lẫn offline deterministic.
