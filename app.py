import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Cấu hình trang
st.set_page_config(page_title="Fraud Detection Test App", layout="wide", page_icon="🚨")

# Tiêu đề ứng dụng
st.title("🚨 Phát hiện Gian lận: Ứng dụng Kiểm tra Mô hình Random Forest")
st.markdown("---")

# Mô tả ứng dụng

st.markdown("---")

# Phần tải lên tệp
st.sidebar.header("📁 Tải lên Mô hình & Dữ liệu")

# Mặc định tên file dựa trên yêu cầu của người dùng
default_model_filename = "best_rf_model_optuna.pkl"
default_data_filename = "fraud_detection_demo_data.csv"

# Sidebar để tải lên file
uploaded_model_file = st.sidebar.file_uploader("1. Tải lên mô hình (.pkl)", type=["pkl"])
uploaded_data_file = st.sidebar.file_uploader("2. Tải lên dữ liệu demo (.csv)", type=["csv"])

# Tùy chọn để sử dụng tên tệp mặc định nếu không có file được tải lên
st.sidebar.markdown("---")
use_defaults = st.sidebar.checkbox("Sử dụng tên tệp mặc định?", value=True)

model = None
data = None

# Thực hiện tải mô hình
if uploaded_model_file is not None:
    with st.spinner("Đang tải mô hình..."):
        try:
            model = joblib.load(uploaded_model_file)
            st.sidebar.success(f"Mô hình được tải lên thành công!")
        except Exception as e:
            st.sidebar.error(f"Lỗi khi tải mô hình: {e}")
elif use_defaults:
    if os.path.exists(default_model_filename):
        try:
            model = joblib.load(default_model_filename)
            st.sidebar.info(f"Đã tải mô hình mặc định: `{default_model_filename}`")
        except Exception as e:
            st.sidebar.error(f"Lỗi khi tải mô hình mặc định: {e}")
    else:
        st.sidebar.warning(f"Mô hình mặc định `{default_model_filename}` không tìm thấy. Hãy chắc chắn tệp nằm cùng thư mục.")

# Thực hiện tải dữ liệu
if uploaded_data_file is not None:
    with st.spinner("Đang tải dữ liệu..."):
        try:
            data = pd.read_csv(uploaded_data_file)
            st.sidebar.success(f"Dữ liệu được tải lên thành công!")
        except Exception as e:
            st.sidebar.error(f"Lỗi khi tải dữ liệu: {e}")
elif use_defaults:
    if os.path.exists(default_data_filename):
        try:
            data = pd.read_csv(default_data_filename)
            st.sidebar.info(f"Đã tải dữ liệu demo mặc định: `{default_data_filename}`")
        except Exception as e:
            st.sidebar.error(f"Lỗi khi tải dữ liệu demo mặc định: {e}")
    else:
        st.sidebar.warning(f"Dữ liệu demo mặc định `{default_data_filename}` không tìm thấy. Hãy chắc chắn tệp nằm cùng thư mục.")

# Phần kiểm tra và dự đoán
if model is not None and data is not None:
    st.header("📊 Kiểm tra Dự đoán trên Dữ liệu Demo")
    st.write(f"Đã tải **{len(data)}** hàng dữ liệu thử nghiệm.")

    # Hiển thị dữ liệu demo
    st.subheader("👀 Dữ liệu Demo (5 hàng đầu tiên)")
    st.dataframe(data.head())

    st.markdown("---")

    # Lựa chọn hàng để dự đoán
    st.subheader("🔍 Chọn một hàng dữ liệu để dự đoán")
    row_to_predict_index = st.number_input("Chọn chỉ số hàng (0 đến N-1):", min_value=0, max_value=len(data)-1, value=0, step=1)
    
    # Hiển thị hàng dữ liệu đã chọn
    selected_row = data.iloc[row_to_predict_index:row_to_predict_index+1]
    st.write(f"Hàng dữ liệu đã chọn (Chỉ số: {row_to_predict_index}):")
    st.dataframe(selected_row)

    # Nút để thực hiện dự đoán
    if st.button("🔮 Thực hiện Dự đoán"):
        with st.spinner("Đang xử lý dự đoán..."):
            try:
                # Tiền xử lý dữ liệu: Loại bỏ cột nhãn mục tiêu (thường là 'Class' hoặc 'Fraud' trong bài toán này)
                # Dựa trên kết quả RF SMOTE, mô hình mong đợi dữ liệu đã được SMOTE xử lý,
                # vì vậy có thể dữ liệu demo vẫn còn cột nhãn mục tiêu.
                # Giả định cột mục tiêu có tên là 'Class' như trong ví dụ.
                target_column_name = "Class" # Cần người dùng xác minh nếu tên cột khác
                
                # Kiểm tra cột mục tiêu
                if target_column_name in data.columns:
                    # Tạo X cho dự đoán bằng cách loại bỏ cột mục tiêu
                    X_to_predict = selected_row.drop(target_column_name, axis=1)
                else:
                    # Nếu không tìm thấy cột mục tiêu, đưa toàn bộ dữ liệu vào (có thể gây lỗi nếu mô hình mong đợi đúng số đặc trưng)
                    X_to_predict = selected_row
                    st.warning(f"Không tìm thấy cột `{target_column_name}`. Giả sử dữ liệu không có nhãn.")

                # Thực hiện dự đoán
                prediction = model.predict(X_to_predict)[0]
                prediction_proba = model.predict_proba(X_to_predict)[0][1] # Xác suất lớp 'Fraud' (1)
                
                # Hiển thị kết quả
                st.subheader("Kết quả Dự đoán của Mô hình Chiến thắng")
                
                # Định nghĩa kết quả dựa trên dự đoán
                fraud_label = "🚨 GIAN LẬN (Gian lận)"
                normal_label = "✅ HỢP LỆ (Hợp lệ)"
                
                # Sử dụng info boxes để hiển thị rõ ràng hơn
                if prediction == 1:
                    st.error(f"Mô hình dự đoán đây là: **{fraud_label}**")
                    st.write(f"**Xác suất Gian lận:** `{prediction_proba:.4f}`")
                else:
                    st.success(f"Mô hình dự đoán đây là: **{normal_label}**")
                    st.write(f"**Xác suất Gian lận:** `{prediction_proba:.4f}`")

            except Exception as e:
                st.error(f"Lỗi khi thực hiện dự đoán: {e}. Có thể dữ liệu demo đầu vào không có đúng số lượng đặc trưng hoặc đúng định dạng mà mô hình mong đợi. Hãy chắc chắn rằng dữ liệu demo đầu vào của bạn có các đặc trưng giống như dữ liệu huấn luyện của mô hình.")

else:
    st.header("Cần tải lên Mô hình và Dữ liệu để Kiểm tra")
    st.info("""
    Hãy chắc chắn rằng:
    1. Bạn đã tải lên một file mô hình `.pkl` hợp lệ (ví dụ: `{0}`).
    2. Bạn đã tải lên dữ liệu demo `.csv` thử nghiệm hợp lệ (ví dụ: `{1}`).
    
    Bạn có thể sử dụng các widget trong sidebar để tải lên các tệp của mình hoặc đánh dấu checkbox để sử dụng các tên tệp mặc định nếu chúng nằm cùng thư mục.
    """.format(default_model_filename, default_data_filename))

# Footer