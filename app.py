import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import os

st.set_page_config(page_title="Password Strength Predictor", page_icon="🔐")

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(password):
    length  = len(password)
    if length == 0:
        return np.zeros(6)
    upper   = sum(1 for c in password if c.isupper())
    lower   = sum(1 for c in password if c.islower())
    digits  = sum(1 for c in password if c.isdigit())
    special = len(re.findall(r'[^a-zA-Z0-9]', password))
    prob    = [password.count(c) / length for c in set(password)]
    entropy = -sum(p * math.log2(p) for p in prob)
    return np.array([length, upper, lower, digits, special, entropy])

# ── Train model (cached — runs once) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Training model…")
def load_model():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    df = pd.read_csv("passwords_dataset.csv")
    df[["length","upper_count","lower_count","digit_count","special_count","entropy"]] = \
        df["Password"].apply(lambda p: pd.Series(extract_features(p)))

    drop_cols = [c for c in ["Password","Has Lowercase","Has Uppercase","Has Special Character","Length"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=["Strength"])
    le = LabelEncoder()
    y  = le.fit_transform(df["Strength"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation="relu", input_shape=(6,)),
        Dense(32, activation="relu"),
        Dense(3,  activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=0)

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    return model, scaler, le, acc

model, scaler, le, acc = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔐 Password Strength Predictor")
st.caption(f"ANN model · Test accuracy: **{acc*100:.1f}%**")
st.divider()

password = st.text_input("Enter a password", type="password", placeholder="Type here…")
show = st.checkbox("👁 Show password")
if show and password:
    st.text(f"🔡 {password}")

if password:
    feats  = extract_features(password)
    scaled = scaler.transform(feats.reshape(1, -1))
    proba  = model.predict(scaled, verbose=0)[0]
    label  = le.inverse_transform([np.argmax(proba)])[0]

    color = {"Weak": "🔴", "Medium": "🟡", "Strong": "🟢"}
    st.subheader(f"{color.get(label, '')} {label}")

    st.write("**Confidence**")
    for cls, p in zip(le.classes_, proba):
        st.progress(float(p), text=f"{cls}: {p*100:.1f}%")

    st.divider()
    st.write("**Feature breakdown**")
    cols = st.columns(6)
    names  = ["Length", "Uppercase", "Lowercase", "Digits", "Special", "Entropy"]
    for col, name, val in zip(cols, names, feats):
        col.metric(name, int(val) if name != "Entropy" else f"{val:.2f}")

    # Tips
    tips = []
    if feats[0] < 8:  tips.append("Use at least 8 characters")
    if feats[1] == 0: tips.append("Add uppercase letters")
    if feats[3] == 0: tips.append("Include numbers")
    if feats[4] == 0: tips.append("Add special characters (!@#$…)")

    if tips:
        st.divider()
        st.write("**💡 Tips to improve**")
        for t in tips:
            st.write(f"• {t}")
