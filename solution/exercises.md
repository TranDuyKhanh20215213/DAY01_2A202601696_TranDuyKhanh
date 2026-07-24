# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> *Khi tăng temperature câu trả lời ngày càng bay bổng và sáng tạo hơn.
> Tuy nhiên tại temperature từ 1.2 trở lên câu trả lời bị kém mạch lạc*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Đối với trợ lý soạn thảo hợp đồng, nên chọn Temperature = 0.0 - 0.2 để bảo đảm tính chuẩn xác tuyệt đối, nhất quán, bám sát mẫu pháp lý và loại bỏ khả năng AI tự "sáng tác" thông tin sai lệch. 
> Ngược lại, với trợ lý viết slogan quảng cáo, nên đặt Temperature = 0.7 - 0.9 nhằm kích thích tính sáng tạo, tạo ra nhiều ý tưởng mới lạ, bất ngờ và đa dạng góc nhìn. 
> Khác biệt cốt lõi: Soạn hợp đồng cần độ tin cậy và chính xác tối đa (ưu tiên các từ ngữ có xác suất đúng cao nhất), trong khi sáng tạo slogan cần độ ngẫu nhiên và phá cách (chấp nhận các liên tưởng từ ngữ ít phổ biến hơn để tạo ấn tượng).*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Ước tính chi phí đầu ra/ngày: Với 20 triệu token đầu ra ($40.000 \text{ lượt} \times 500 \text{ tokens}$), GPT-4o tốn 200\$/ngày (~6.000\$/tháng), trong khi GPT-4o-mini chỉ tốn 12\$/ngày (~360\$/tháng) — chênh lệch gần 17 lần. 
> Dùng model lớn (gpt-4o): Khi làm các tác vụ suy luận phức tạp, phân tích dữ liệu đa bước hoặc tư vấn chuyên sâu (chẩn đoán y tế, tư vấn pháp lý) — nơi chất lượng câu trả lời quyết định độ an toàn của hệ thống. 
> Dùng model nhỏ (gpt-4o-mini): Khi làm các tác vụ định hình đơn giản, phân loại dữ liệu, sửa lỗi văn bản hoặc CSKH cơ bản — nơi model nhỏ xử lý đủ tốt nhưng tiết kiệm tối đa ngân sách.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Phản hồi của nhà thơ sử dụng giọng văn lãng mạn, nhịp điệu giàu hình ảnh ví von, độ dài ngắn gọn và không chứa thuật ngữ kỹ thuật.
> Trong khi đó, phản hồi của kỹ sư phần mềm senior mang phong cách trang trọng, cấu trúc chặt chẽ, sử dụng các khái niệm kỹ thuật . Từ đó rút ra System Prompt có thể điều khiển trực tiếp: (1) Persona/Giọng văn & thái độ, (2) Đối tượng mục tiêu & độ sâu kỹ thuật, (3) Định dạng đầu ra (cấu trúc trình bày hay có kèm mã code hay không).*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Với đoạn văn tiếng Việt 164 từ, ước lượng thô `150 / 0.75` cho kết quả 218 token, nhưng `tiktoken` thực tế trả về 184 token (chênh lệch 15.6\%). Nếu dùng ước lượng thô `từ / 0.75`, thì sẽ dự đoán THỪA vì các tokenizer hiện nay đã hiện đại và tối ưu lên rất nhiều, tiếng việt không còn tiêu tốn quá nhiều token.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Chatbot văn bản (a) hưởng lợi nhiều nhất từ streaming vì giảm tối đa thời gian phản hồi ban đầu (TTFT - Time To First Token), giúp người dùng đọc nội dung ngay lập tức thay vì nhìn màn hình trống vài giây. Trợ lý giọng nói (b) cũng hưởng lợi từ streaming để bắt đầu tổng hợp tiếng nói (TTS) ngay từ những câu/cụm từ đầu tiên. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm (c) hoàn toàn không cần streaming vì là tác vụ xử lý hàng loạt (batch job) không tương tác real-time với người dùng; việc nhận từng chunk vừa gây tốn tài nguyên mạng vừa làm tăng độ phức tạp khi ghép nối tài liệu.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Exponential backoff kéo giãn khoảng thời gian giữa các lần retry theo cấp số nhân (ví dụ: 0.1s -> 0.2s -> 0.4s -> 0.8s), giúp giảm mật độ yêu cầu dồn dập và tạo khoảng nghỉ cho phía server hạ nhiệt/hồi phục, tránh hiện tượng quá tải dây chuyền so với delay cố định. Kỹ thuật "jitter" (thêm nhiễu thời gian ngẫu nhiên vào delay) giải quyết vấn đề "Thảm họa đồng bộ" (Thundering Herd Problem / Retry Storm), vốn xảy ra khi hàng nghìn client bị lỗi cùng lúc và retry đồng loạt tại đúng các mốc thời gian backoff giống hệt nhau.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *System Prompt: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."
> 
> Hai chỗ quan trọng nếu xóa đi sẽ làm thay đổi hành vi:
> 1. Xóa "trả lời ngắn gọn": Trợ lý sẽ chuyển sang trả lời dài dòng, giải thích lan man nhiều đoạn văn gây tốn token và làm chậm tốc độ phản hồi.
> 2. Xóa "trợ giảng thân thiện": Trợ lý sẽ mất đi giọng văn gần gũi, cởi mở và phong cách khuyến khích học tập, quay về giọng điệu trung tính, máy móc hoặc cứng nhắc của model mặc định.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Tình huống: Người dùng khai báo ngữ cảnh ở lượt 1 ("Tôi đang làm dự án e-commerce bán quần áo trẻ em tại Hà Nội"). Sau đó trải qua 5 lượt trao đổi chi tiết về giao diện và thanh toán. Ở lượt thứ 7, người dùng hỏi "Chiến dịch marketing nào phù hợp với sản phẩm của tôi?". Do history chỉ giữ 4 lượt gần nhất (lượt 3 đến 6), thông tin "bán quần áo trẻ em tại Hà Nội" ở lượt 1 đã bị xóa, khiến trợ lý không biết sản phẩm là gì và đưa ra tư vấn chung chung.
> 
> Cách khắc phục: Sử dụng kỹ thuật "Context Summarization" (Tóm tắt ngữ cảnh). Khi history vượt quá 4 lượt, thay vì xóa bỏ hoàn toàn các lượt cũ, sử dụng model nhỏ (như gpt-4o-mini) để tóm tắt các ý chính/thông tin quan trọng của cuộc trò chuyện thành một đoạn "System Memory" và đính kèm ở đầu history.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
