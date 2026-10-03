# Bước 8: Phân Tích Kết Quả Benchmark & Trade-off Kỹ Thuật

Tài liệu này tổng hợp kết quả chạy thực tế và phân tích chi tiết hành vi của hai kiến trúc agent (**Baseline Agent** vs **Advanced Agent**) trên bộ dữ liệu benchmark tiếng Việt theo yêu cầu của **Bước 8** trong `Guide.md`.

---

## 1. Bảng Kết Quả Benchmark

### 1.1. Standard Benchmark (`data/conversations.json`)
Bộ dữ liệu gồm 10 phiên hội thoại thông thường của người dùng `dungct`, mỗi phiên có các câu hỏi kiểm tra khả năng nhớ chéo phiên (`recall_questions`) được hỏi ở một thread hoàn toàn mới.

| Agent Name | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 2209 | 17356 | **2%** | 0.31 | 0 | 0 |
| **Advanced Agent** | 1769 | 22093 | **100%** | 1.00 | 282 | 0 |

### 1.2. Long-Context Stress Benchmark (`data/advanced_long_context.json`)
Bộ dữ liệu stress test gồm 1 phiên hội thoại dài 16 lượt dày đặc thông tin tin tức, cập nhật địa điểm và nghề nghiệp của người dùng `dungct_stress`.

| Agent Name | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| **Baseline Agent** | 218 | 21725 | **0%** | 0.30 | 0 | 0 |
| **Advanced Agent** | 531 | **13554** | **100%** | 1.00 | 242 | **26** |

---

## 2. Phân Tích Chi Tiết 4 Câu Hỏi Trọng Tâm

### 2.1. Vì sao Advanced Agent có Recall vượt trội so với Baseline Agent?
- **Cơ chế của Baseline Agent:** Chỉ lưu trữ lịch sử tin nhắn trong phạm vi của từng `thread_id` (short-term thread memory). Khi chuyển sang một `thread_id` mới (các câu hỏi `recall_questions` được đặt trong một phiên mới), Baseline Agent không mang theo bất kỳ ngữ cảnh nào từ quá khứ, dẫn tới việc quên sạch thông tin (`Cross-session recall` chỉ đạt **0% - 2%**).
- **Cơ chế của Advanced Agent:** Được trang bị lớp **Persistent Memory** với file `User.md` thông qua `UserProfileStore`. Ngay khi người dùng đề cập đến các thông tin ổn định (tên, nơi ở, nghề nghiệp, đồ uống yêu thích, sở thích kỹ thuật...), agent tự động trích xuất và lưu bền vững vào `state/profiles/<user_id>.md`. Dù mở thread mới, `User.md` vẫn luôn được tải vào prompt context, giúp Advanced Agent duy trì **Recall 100%** tuyệt đối.

### 2.2. Vì sao Advanced Agent có thể tốn hơn ở hội thoại ngắn?
- Quan sát ở **Standard Benchmark**, tổng `Prompt tokens processed` của Advanced Agent là **22,093 tokens**, cao hơn mức **17,356 tokens** của Baseline Agent (chênh lệch khoảng 27%).
- **Nguyên nhân:** Để duy trì bộ nhớ dài hạn, ở mỗi lượt hội thoại, Advanced Agent đều phải kéo theo nội dung của file `User.md` vào prompt context (`md_tokens`). Trong các cuộc hội thoại ngắn (khi dung lượng tin nhắn chưa vượt ngưỡng nén), chi phí đọc file profile liên tục tạo ra một khoản overhead cố định (fixed cost per turn).

### 2.3. Vì sao Compact Memory giúp Advanced Agent có lợi thế áp đảo ở hội thoại dài?
- Quan sát ở **Long-Context Stress Benchmark**:
  - `Baseline Agent` tích lũy toàn bộ 16 lượt tin nhắn dài không qua cắt lọc, khiến `Prompt tokens processed` tăng vọt lên **21,725 tokens**.
  - `Advanced Agent` kích hoạt cơ chế **Compact Memory** tới **26 lần**, nén các lượt hội thoại cũ thành bản tóm tắt gọn gàng và chỉ giữ lại một số lượng nhỏ tin nhắn gần nhất (`compact_keep_messages`).
- **Kết quả:** `Prompt tokens processed` của Advanced Agent giảm mạnh xuống còn **13,554 tokens** (tiết kiệm **37.6% chi phí ngữ cảnh** so với Baseline). Đồng thời, việc nén không hề làm mất fact quan trọng vì các thông tin cốt lõi đã được bảo toàn trong `User.md`.

### 2.4. File Memory tăng trưởng ra sao và rủi ro gì đi kèm?
- **Tốc độ tăng trưởng:**
  - File `User.md` chỉ tăng khoảng **242 - 282 bytes** cho mỗi người dùng sau hàng chục lượt chat nhờ định dạng Markdown có cấu trúc (`- Key: Value`).
- **Các rủi ro kỹ thuật đi kèm:**
  1. **Nhiễu thông tin (Noise & Trivia Extraction):** Nếu không có ngưỡng tin cậy (`confidence threshold`), những câu bông đùa hoặc thông tin tạm thời (ví dụ: *"đùa là làm product manager"*, *"Hà Nội chỉ là nơi đi họp 2 ngày"*) rất dễ bị lưu nhầm thành fact dài hạn.
  2. **Mâu thuẫn thông tin (Conflict Handling):** Khi người dùng thay đổi thông tin (ví dụ: từ Đà Nẵng sang Huế rồi lại sang Đà Nẵng; từ Backend sang MLOps), nếu hệ thống chỉ append mà không có cơ chế `upsert` hoặc xóa bỏ fact cũ, agent sẽ gặp hiện tượng ảo giác (hallucination) hoặc trả về thông tin mâu thuẫn.
  3. **Memory Phình To & Chi phí ngữ cảnh dài hạn:** Khi người dùng tương tác qua nhiều tháng/năm, nếu không có cơ chế phân rã (`memory decay`) hoặc dọn dẹp định kỳ, file `User.md` sẽ dần trở nên quá lớn, làm giảm tốc độ xử lý và tăng chi phí token ở mọi lượt gọi.

---

## 3. Mở Rộng Kỹ Thuật Đạt Điểm Cao (Bonus Features)

Hệ thống đã triển khai các cơ chế guardrail nâng cao:
1. **Structured Entity Extraction & Upsert:** Quản lý facts theo từng trường chuẩn hóa (`Name`, `Location`, `Profession`, `Response Style`, `Favorite Drink`, `Favorite Food`, `Pet`, `Tech Interests`).
2. **Conflict Resolution:** Nhận diện các tín hiệu sửa đổi (`đính chính`, `chuyển sang`, `giờ mình ở...`) để ghi đè dữ liệu cũ, không lưu trữ đồng thời hai giá trị mâu thuẫn.
3. **Question-filtering (Confidence Guardrail):** Bỏ qua các lượt câu hỏi thuần túy (`recall_questions`), ngăn chặn việc ghi dữ liệu rác vào bộ nhớ hồ sơ người dùng.
