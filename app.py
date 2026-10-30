
import streamlit as st
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline


st.set_page_config(
    page_title="Dự đoán giá nhà",
    layout="centered"
)

TRAIN_FILE = "Nhom2_GiaNha_train.xlsx"

FEATURES = [
    "BedroomAbvGr",
    "FullBath",
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "OverallQual"
]

TARGET = "SalePrice"


@st.cache_resource
def load_models():

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            "Không tìm thấy file dữ liệu huấn luyện: "
            + TRAIN_FILE
        )

    data = pd.read_excel(TRAIN_FILE)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "File Excel thiếu các cột: "
            + ", ".join(missing_columns)
        )

    data = data[required_columns].copy()

    for col in required_columns:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.dropna()

    if len(data) < 10:
        raise ValueError(
            "Dữ liệu huấn luyện không đủ."
        )

    X = data[FEATURES]
    y = data[TARGET]

    ann_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42
        ))
    ])

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=100,
            gamma="scale"
        ))
    ])

    ann_model.fit(X, y)
    svm_model.fit(X, y)

    return {
        "ANN": ann_model,
        "SVM": svm_model
    }


st.title("DỰ ĐOÁN GIÁ NHÀ")

st.write(
    "Chọn loại nhà và nhập thông số để dự đoán giá nhà."
)


# Load models
try:
    models = load_models()

except Exception as e:
    st.error(str(e))
    st.stop()


# Step 1: Select house type first
st.subheader("BƯỚC 1: CHỌN LOẠI NHÀ")

house_type = st.selectbox(
    "Bạn muốn tìm giá loại nhà nào?",
    [
        "Nhà 1 tầng",
        "Nhà 2 tầng",
        "Nhà không có tầng 1 và tầng 2"
    ],
    key="house_type"
)


# Step 2: Enter house information
st.subheader("BƯỚC 2: NHẬP THÔNG SỐ NHÀ")

bedroom = st.number_input(
    "Số phòng ngủ",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

fullbath = st.number_input(
    "Số phòng tắm",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

lotarea = st.number_input(
    "Diện tích đất",
    min_value=1,
    value=8000,
    step=100
)

yearbuilt = st.number_input(
    "Năm xây dựng",
    min_value=1800,
    max_value=2026,
    value=2000,
    step=1
)

overallqual = st.slider(
    "Chất lượng xây dựng",
    min_value=1,
    max_value=10,
    value=5
)


# Floor area depends on house type
firstflrsf = 0
secondflrsf = 0

if house_type == "Nhà 1 tầng":

    firstflrsf = st.number_input(
        "Diện tích tầng 1",
        min_value=0,
        max_value=int(lotarea),
        value=min(1000, int(lotarea)),
        step=50
    )

elif house_type == "Nhà 2 tầng":

    firstflrsf = st.number_input(
        "Diện tích tầng 1",
        min_value=0,
        max_value=int(lotarea),
        value=min(1000, int(lotarea)),
        step=50
    )

    secondflrsf = st.number_input(
        "Diện tích tầng 2",
        min_value=0,
        max_value=int(lotarea),
        value=min(500, int(lotarea)),
        step=50
    )

else:

    st.write(
        "Loại nhà này không có tầng 1 và tầng 2."
    )


# Step 3: Select model
st.subheader("BƯỚC 3: CHỌN MÔ HÌNH")

model_name = st.selectbox(
    "Mô hình dự đoán",
    ["ANN", "SVM"]
)


# Step 4: Predict
if st.button("DỰ ĐOÁN GIÁ NHÀ", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            bedroom,
            fullbath,
            lotarea,
            yearbuilt,
            firstflrsf,
            secondflrsf,
            overallqual
        ]],
        columns=FEATURES
    )

    prediction = models[model_name].predict(
        input_data
    )[0]

    st.subheader("KẾT QUẢ DỰ ĐOÁN")

    st.write("Loại nhà:", house_type)
    st.write("Mô hình:", model_name)

    st.success(
        "Giá nhà dự đoán: {:,.0f}".format(prediction)
    )
