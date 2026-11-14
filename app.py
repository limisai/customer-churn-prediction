import streamlit as st

from predict import load_model, predict_customer


LABELS = {
    "gender": "Пол",
    "SeniorCitizen": "Пожилой клиент",
    "Partner": "Есть партнёр",
    "Dependents": "Есть иждивенцы",
    "PhoneService": "Телефонная связь",
    "MultipleLines": "Несколько телефонных линий",
    "InternetService": "Тип интернета",
    "OnlineSecurity": "Защита в интернете",
    "OnlineBackup": "Резервное копирование",
    "DeviceProtection": "Защита устройств",
    "TechSupport": "Техническая поддержка",
    "StreamingTV": "Потоковое ТВ",
    "StreamingMovies": "Потоковое кино",
    "Contract": "Тип договора",
    "PaperlessBilling": "Электронные счета",
    "PaymentMethod": "Способ оплаты",
}
TRANSLATIONS = {
    "Yes": "Да", "No": "Нет", "Female": "Женский", "Male": "Мужской",
    "Month-to-month": "Помесячный", "One year": "Годовой", "Two year": "Двухлетний",
    "Fiber optic": "Оптоволокно", "No internet service": "Нет интернета",
    "No phone service": "Нет телефонной связи",
    "Bank transfer (automatic)": "Автоперевод из банка",
    "Credit card (automatic)": "Автоплатёж по карте",
    "Mailed check": "Чек по почте", "Electronic check": "Электронный чек",
}


@st.cache_resource
def trained_model():
    return load_model()


def main():
    st.set_page_config(page_title="Прогноз оттока", layout="wide")
    st.title("Прогноз оттока клиента")
    st.caption("Логистическая регрессия, обученная на синтетических данных")
    try:
        artifact = trained_model()
    except FileNotFoundError:
        st.error("Модель не найдена. Создайте её локально командой `python main.py`.")
        st.stop()

    preprocessing = artifact["pipeline"].named_steps["preprocessing"]
    encoder = preprocessing.named_transformers_["categorical"]
    categories = dict(zip(encoder.feature_names_in_, encoder.categories_, strict=True))

    def select(customer, name):
        customer[name] = st.selectbox(
            LABELS[name], categories[name], format_func=lambda value: TRANSLATIONS.get(value, value)
        )

    with st.form("customer"):
        customer = {}
        profile, billing, services = st.columns(3)
        with profile:
            st.subheader("Клиент")
            for name in ("gender", "Partner", "Dependents"):
                select(customer, name)
            customer["SeniorCitizen"] = int(st.checkbox(LABELS["SeniorCitizen"]))
            customer["tenure"] = st.number_input("Стаж, месяцев", min_value=0, value=6)
        with billing:
            st.subheader("Договор и оплата")
            for name in ("Contract", "PaperlessBilling", "PaymentMethod"):
                select(customer, name)
            customer["MonthlyCharges"] = st.number_input("Ежемесячная плата", min_value=0.0, value=66.17)
            customer["TotalCharges"] = st.number_input(
                "Накопленная плата", min_value=0.0, value=None, placeholder="Можно оставить пустым"
            )
        with services:
            st.subheader("Услуги")
            for name in ("PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                         "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"):
                select(customer, name)
        submitted = st.form_submit_button("Рассчитать вероятность", type="primary")

    if submitted:
        result = predict_customer(artifact, customer)
        probability = result["probability_churn_yes"]
        threshold = result["threshold"]
        st.subheader("Результат")
        probability_card, label_card, threshold_card = st.columns(3)
        probability_card.metric("Вероятность оттока", f"{probability:.1%}")
        label_card.metric("Класс Churn", result["churn"])
        threshold_card.metric("Порог", f"{threshold:.2g}")
        st.progress(probability, text="Вероятность оттока по шкале от 0 до 100%")
        if result["churn"] == "Yes":
            st.warning("Вероятность достигла порога оттока.")
        else:
            st.success("Вероятность ниже порога оттока.")


if __name__ == "__main__":
    main()
