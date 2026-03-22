# Phát hiện Gian lận Thẻ Tín dụng

## Mô tả

Dự án này triển khai một ứng dụng web sử dụng mô hình học máy Random Forest để phát hiện gian lận thẻ tín dụng. Ứng dụng được xây dựng bằng Streamlit, cho phép người dùng tải lên mô hình và dữ liệu demo để kiểm tra và dự đoán gian lận.

## Tính năng

- **Tải lên mô hình**: Hỗ trợ tải lên file mô hình (.pkl) hoặc sử dụng mô hình mặc định.
- **Tải lên dữ liệu**: Tải lên dữ liệu demo (.csv) để kiểm tra.
- **Dự đoán gian lận**: Nhập dữ liệu giao dịch và nhận kết quả dự đoán.
- **Giao diện thân thiện**: Giao diện web đơn giản, dễ sử dụng.

## Cài đặt

1. **Clone repository**:
   ```bash
   git clone <repository-url>
   cd MLP305
   ```

2. **Tạo môi trường ảo**:
   ```bash
   python -m venv MLP305
   ```

3. **Kích hoạt môi trường ảo**:
   ```bash
   source MLP305/bin/activate  # Trên Linux/Mac
   # hoặc MLP305\Scripts\activate trên Windows
   ```

4. **Cài đặt các thư viện yêu cầu**:
   ```bash
   pip install -r requirements.txt
   ```

## Sử dụng

1. **Chạy ứng dụng**:
   ```bash
   streamlit run app.py
   ```

2. **Mở trình duyệt**: Ứng dụng sẽ chạy trên `http://localhost:8501`.

3. **Sử dụng ứng dụng**:
   - Tải lên mô hình (.pkl) và dữ liệu demo (.csv) qua sidebar.
   - Hoặc chọn sử dụng tên tệp mặc định.
   - Nhập dữ liệu giao dịch để dự đoán gian lận.

## Dữ liệu

- **Dataset chính**: `creditcard.csv` - Dữ liệu giao dịch thẻ tín dụng.
- **Dữ liệu demo**: `fraud_detection_demo_data.csv` - Dữ liệu mẫu để kiểm tra.

## Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- Các thư viện: streamlit, pandas, joblib, scikit-learn, numpy

## Cấu trúc dự án

- `app.py`: File chính của ứng dụng Streamlit.
- `fraud_detection_model.ipynb`: Notebook Jupyter cho việc huấn luyện mô hình.
- `requirements.txt`: Danh sách thư viện cần thiết.
- `creditcard.csv`: Dataset gốc.
- `fraud_detection_demo_data.csv`: Dữ liệu demo.
- `.gitignore`: Bỏ qua các file không cần thiết (model, CSV, môi trường ảo).

## Đóng góp

Nếu bạn muốn đóng góp, vui lòng tạo pull request hoặc báo cáo issue.

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file LICENSE để biết thêm chi tiết.