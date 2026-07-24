import sys
from template import count_tokens, OPENAI_MODEL

TEXT = """Trong cuộc sống, thành công không phải là điểm đến đạt được chỉ sau một đêm, mà là kết quả của một quá trình kiên trì không ngừng nghỉ. Mỗi người chúng ta đều phải đối mặt với vô vàn khó khăn, thử thách và cả những thất bại đắng ngắt. Nhưng chính vấp ngã mới là người thầy tốt nhất, dạy ta cách đứng dậy mạnh mẽ hơn. Sự kiên trì giống như giọt nước chảy mãi cũng làm mòn đá. Nó không yêu cầu bạn phải có bước nhảy vọt phi thường, mà chỉ cần bạn bền bỉ tiến về phía trước mỗi ngày, dù chỉ là những bước đi nhỏ nhất. Khi bạn không bỏ cuộc, mọi rào cản rồi sẽ trở thành bệ phóng đưa bạn đến gần hơn với mục tiêu. Hãy tin tưởng vào hành trình của chính mình, bởi vì gieo hạt kiên nhẫn hôm nay chắc chắn sẽ thu hoạch được quả ngọt vào ngày mai."""

if __name__ == "__main__":
    # 1. Đếm số từ
    word_count = len(TEXT.split())

    # 2. Đếm số token bằng hàm count_tokens từ template.py
    token_count = count_tokens(TEXT, model=OPENAI_MODEL)

    # 3. Ước lượng thô theo công thức Part 1 (số từ / 0.75)
    rough_estimate = int(word_count / 0.75)

    print("=" * 60)
    print("KẾT QUẢ THỐNG KÊ TOKEN ĐOẠN VĂN")
    print("=" * 60)
    print(f"Model sử dụng            : {OPENAI_MODEL}")
    print(f"Tổng số ký tự             : {len(TEXT)}")
    print(f"Tổng số từ (word count)   : {word_count}")
    print(f"Số token chính xác       : {token_count}")
    print(f"Ước lượng thô (từ / 0.75) : ~{rough_estimate}")
    print(f"Chênh lệch thực tế        : +{token_count - rough_estimate} token ({((token_count - rough_estimate) / rough_estimate) * 100:.1f}%)")
    print("=" * 60)
