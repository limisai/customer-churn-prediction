# Прогнозирование оттока клиентов

Проект сравнивает две модели логистической регрессии для прогнозирования оттока клиентов телеком-оператора. Такой прогноз помогает команде по удержанию выбрать, с кем связаться в первую очередь. Пропустить клиента, который собирается уйти, и обратиться к клиенту, который и так останется, — ошибки с разной стоимостью.

## Данные

Проект использует **искусственный датасет** `data/synthetic/synthetic_customer_churn.csv`. В нём 7 043 вымышленных клиента, из которых 2 091 получили метку оттока (29,7%). Генератор не читает исходный датасет и создаёт идентификаторы вида `SYN-00001`.

`generate_dataset.py` использует NumPy и seed 42. У клиентов с большим стажем чаще встречаются долгосрочные договоры. Подключённые услуги и скидки за договор определяют ежемесячную плату. Накопленная плата зависит от стажа и ежемесячной платы с поправкой на изменение тарифов. После генерации признаков скрипт вычисляет вероятность оттока по стажу, договору, платежам и услугам с добавлением шума. Затем он выбирает метку `Churn` по этой вероятности.

## Методика

Проект использует Python, pandas, NumPy и scikit-learn.

1. Исключаем `customerID` и отделяем целевую переменную от признаков. Преобразуем `TotalCharges` в числовой столбец. Пустые и некорректные значения считаем пропусками.
2. Делим данные на обучающую и тестовую выборки в пропорции 80/20 со стратификацией по `Churn` и `random_state=42`. В обучающей выборке 5 634 строки, в тестовой — 1 409.
3. Заполняем числовые пропуски нулями и стандартизируем числовые признаки. Генератор сохраняет нулевые накопленные суммы при нулевом стаже и не добавляет пропусков.
4. Кодируем категории с помощью `OneHotEncoder`. Для незнакомой категории все закодированные столбцы соответствующего признака получают нули. Обучаем преобразования только на обучающей выборке и применяем их к тестовой.
5. Обучаем две логистические регрессии на одинаковых данных. В базовой модели все наблюдения имеют одинаковый вес. В модели с балансировкой веса классов обратно пропорциональны их частотам в обучающей выборке.

## Результаты

Обе модели оценены на одной тестовой выборке. Precision, recall и F1 рассчитаны для класса оттока (`Yes`).

| Модель | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Без весов классов | 0.7530 | 0.6190 | 0.4354 | 0.5112 |
| С балансировкой | 0.7019 | 0.4984 | 0.7512 | 0.5992 |

В матрицах ошибок положительный класс — `Yes`. При запуске скрипта строки соответствуют фактическим классам, столбцы — предсказанным. Порядок по обеим осям — `No`, `Yes`.

| Модель | Верно предсказали отсутствие оттока | Ложные срабатывания | Пропущенные случаи оттока | Верно обнаружили отток |
| --- | ---: | ---: | ---: | ---: |
| Без весов классов | 879 | 112 | 236 | 182 |
| С балансировкой | 675 | 316 | 104 | 314 |

Модель с балансировкой обнаружила на 132 случая оттока больше и повысила F1 с 0.5112 до 0.5992. Ложных срабатываний стало на 204 больше. Приоритет охвата благоприятствует модели с балансировкой, а ограниченный бюджет на контакты — модели без весов.

## Запуск

Выполняй команды из корня проекта. Эксперимент проверен на Python 3.13.7. Синтетический CSV входит в репозиторий. Строка с генератором позволяет воспроизвести этот файл.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe generate_dataset.py
.venv\Scripts\python.exe main.py
```

На macOS или Linux в последних трёх командах используй `.venv/bin/python`.

## Ограничения

- Результаты одного тестового разбиения показывают, насколько хорошо модели улавливают зависимости, заданные генератором. Они не подтверждают качество прогноза на реальных клиентах или будущих периодах.
- Сравнение использует стандартное правило предсказания каждой модели. Для выбора порога срабатывания нужны валидационная выборка и оценки стоимости пропущенного оттока и лишнего контакта.

---

# Customer Churn Prediction

This project compares two logistic regression models for predicting whether a telecom customer will churn. A retention team could use churn predictions to prioritize outreach. Missing an at-risk customer and contacting a customer who would stay have different costs, so the comparison reports both types of error.

## Dataset

The project uses an **artificial dataset** at `data/synthetic/synthetic_customer_churn.csv`. It contains 7,043 fictional customers, including 2,091 labeled as churned (29.7%). The generator creates identifiers such as `SYN-00001` and reads no source records.

`generate_dataset.py` uses NumPy with seed 42. Longer-tenure customers are more likely to have long-term contracts. Subscribed services and contract discounts determine monthly charges. Total charges scale with tenure and monthly charges, with variation to approximate historical billing changes. After building the features, the generator calculates churn probability from tenure, contract, charges, and services, with an independent noise term. It then samples the `Churn` label from that probability.

## Method

The project uses Python, pandas, NumPy, and scikit-learn.

1. Exclude `customerID` and separate the target from the features. Parse `TotalCharges` as numeric, treating blanks and invalid entries as missing.
2. Create a stratified 80/20 train/test split with `random_state=42` (5,634 training rows and 1,409 test rows).
3. Fill numeric missing values with zero if present and standardize numeric features. The generated dataset has no missing values and uses a total charge of zero for zero-tenure customers.
4. One-hot encode categorical features. An unseen category produces zeros for that feature's encoded columns. Fit all preprocessing on training data and reuse the fitted transformations on test data.
5. Train two logistic regression models on the same processed training data. The unweighted baseline gives observations equal weight. The balanced model weights classes inversely to their training frequencies.

## Results

Both models use the same held-out test set. Precision, recall, and F1 are reported for churn (`Yes`).

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Unweighted | 0.7530 | 0.6190 | 0.4354 | 0.5112 |
| Balanced | 0.7019 | 0.4984 | 0.7512 | 0.5992 |

Confusion matrix counts use `Yes` as the positive class. The script prints matrices with actual classes as rows and predicted classes as columns, both in `No`, `Yes` order.

| Model | True negatives | False positives | False negatives | True positives |
| --- | ---: | ---: | ---: | ---: |
| Unweighted | 879 | 112 | 236 | 182 |
| Balanced | 675 | 316 | 104 | 314 |

The balanced model detected 132 more churn cases and increased F1 from 0.5112 to 0.5992, with 204 additional false positives. It is the better candidate when recall matters most. The unweighted model has higher precision and flags fewer customers for outreach.

## Run locally

Run these commands from the project root. The experiment was verified with Python 3.13.7. The synthetic CSV is included in the repository. Run the generator to reproduce it.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe generate_dataset.py
.venv\Scripts\python.exe main.py
```

On macOS or Linux, use `.venv/bin/python` in the last three commands.

## Limitations

- A single holdout split measures how well the models recover relationships built into the simulator. These metrics do not establish performance on real customers or future periods.
- The comparison uses each model's default decision rule. Choosing an outreach threshold would require validation data and estimates of the cost of missed churn and unnecessary contact.
