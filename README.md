# [HCMUS] Master 2024 - LLM course - Final project

## Description

Source code cho đồ án hệ thống chatbot hỏi đáp (Quesion Answering), hỗ trợ đọc báo.

Trong repository này chứa source code cần thiết để deploy lại được local app của hệ thống chatbot.

Ngoài ra, còn có source code cho các tác vụ khác, độc lập với pipeline chạy của app.  Cụ thể là `crawling`, `finetune`, `evaluate` nằm trong các thư mục tương ứng, cùng tên (`\misc\` + TASK_NAME + `_source\`).

### Functionality
- [x] Trả lời câu hỏi
- [x] Xây dựng database từ crawling các bài báo trong ngày
- [x] Tự sinh câu hỏi follow-up
- [x] Xuất nguồn URL của thông tin được sử dụng khi trả lời
- [x] Thu thập feedback của người dùng
- [ ] Chức năng tự cải thiện mô hình, sử dụng dữ liệu trên - Dừng ở mức finetune

## Getting Started

### Dependencies

Cài đặt các packages cần thiết để deploy app chatbot 
```
pip install -r .\requirements.txt
``` 

Ngoài ra, các tác vụ như `crawling`, `finetune`, `evaluate` có file `.\requirements.txt` riêng trong thư mục tương ứng.

### Executing program

Thực thi powershell script file `start.ps1`
```
powershell [-noexit] -executionpolicy bypass -File .\start.ps1
```

Hoặc khởi động thư viện streamlit và trực tiếp thực thi câu lệnh:
```
streamlit run source/app.py
```

## Authors
* Huỳnh Lâm Hải Đăng - 23C15024
* Khấu Đặng Nhật Minh - 23C15032
* Nguyễn Quốc Khánh Tuyên - 23C15040
* Lê Trường Vũ - 23C15042
* Nguyễn Ngọc Thiện - 23C15043

## License

This project is licensed under the **MPL-2.0 License** - see the LICENSE file for details

## Acknowledgments

Chúng em cảm ơn thầy Nguyễn Tiến Huy và thầy Lê Thanh Tùng với sự hướng dẫn và nhận xét cho đồ án của nhóm.