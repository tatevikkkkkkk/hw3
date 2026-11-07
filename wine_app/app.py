import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

st.set_page_config(page_title='Подбор хороших вин', page_icon='🍷', layout='wide')


# ---------- Кэширование (задание 4.5) ----------
# st.cache_resource — для "ресурсов": объект создаётся один раз и общий для всех
# пользователей и перезапусков скрипта, не копируется и не сериализуется. Модель — как раз
# такой тяжёлый неизменяемый объект. st.cache_data — для данных (DataFrame, массивы):
# результат сериализуется и каждому вызову отдаётся копия, поэтому случайное изменение
# таблицы не испортит кэш. Если вызвать joblib.load без декоратора, модель будет
# заново читаться с диска при КАЖДОМ движении слайдера (скрипт выполняется сверху вниз).

@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    # _model не хешируется; вероятности считаются один раз, от порога они не зависят
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
test = load_test()
X_test, y_test = test.drop(columns='target'), test['target'].values
FEATURES = list(X_test.columns)
p_test = test_proba(model, X_test)

# ---------- Сайдбар: порог ----------
st.sidebar.header('Настройки')
threshold = st.sidebar.slider('Порог классификации', 0.0, 1.0,
                              float(config['threshold']), 0.01, key='threshold')
st.sidebar.caption(f'Порог по умолчанию из config.json: {config["threshold"]:.2f}')

st.title('🍷 Отбор хороших вин в подборку')
st.write('Модель логистической регрессии по результатам лабораторного анализа оценивает '
         'вероятность того, что вино **хорошее** (оценка экспертов ≥ 7).')

tab1, tab2, tab3, tab4 = st.tabs(['Оценка одного вина', 'Модельные метрики',
                                  'Пакетное предсказание', 'Что выучила модель'])

# ---------- 4.1 Оценка одного вина ----------
with tab1:
    st.subheader('Введите характеристики вина')
    values = {}
    cols = st.columns(3)
    num_features = [f for f in FEATURES if f != 'is_red']
    for i, f in enumerate(num_features):
        col = X_test[f]
        lo, hi, med = float(col.min()), float(col.max()), float(col.median())
        step = float(10 ** np.floor(np.log10((hi - lo) / 100))) if hi > lo else 0.01
        values[f] = cols[i % 3].number_input(f, min_value=lo, max_value=hi, value=med,
                                             step=step, format='%.4f')
    wine_type = st.radio('Тип вина', ['белое', 'красное'], horizontal=True)
    values['is_red'] = 1 if wine_type == 'красное' else 0

    x = pd.DataFrame([values])[FEATURES]
    proba = model.predict_proba(x)[0, 1]
    c1, c2 = st.columns(2)
    c1.metric('Вероятность, что вино хорошее', f'{proba:.1%}')
    if proba >= threshold:
        c2.success(f'Решение: ✅ берём в подборку (порог {threshold:.2f})')
    else:
        c2.error(f'Решение: ❌ не берём в подборку (порог {threshold:.2f})')

# ---------- 4.2 Модельные метрики ----------
with tab2:
    st.subheader(f'Качество на тестовой выборке при пороге {threshold:.2f}')
    pred = (p_test >= threshold).astype(int)
    prec = metrics.precision_score(y_test, pred, zero_division=0)
    rec = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Precision', f'{prec:.3f}')
    m2.metric('Recall', f'{rec:.3f}')
    m3.metric('F1', f'{f1:.3f}')
    m4.metric('Вин в подборке', int(pred.sum()))

    st.info('**Для сомелье.** Если **увеличить** порог, в подборку попадёт меньше вин, но среди них '
            'будет больше действительно хороших (precision растёт), зато часть хороших вин будет '
            'пропущена (recall падает). Если **уменьшить** порог — подборка станет шире и почти все '
            'хорошие вина попадут в неё, но вырастет доля посредственных.')

    fig_cm, fig_roc, fig_pr = st.columns(3)

    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['не хорошее', 'хорошее']).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Матрица ошибок')
    fig_cm.pyplot(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    tn, fp, fn, tp = cm.ravel()
    cur_fpr = fp / (fp + tn) if fp + tn else 0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, p_test):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.scatter(cur_fpr, rec, color='red', s=60, zorder=5, label='текущий порог')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR (recall)')
    ax.set_title('ROC-кривая')
    ax.legend(loc='lower right')
    fig_roc.pyplot(fig)

    pr_p, pr_r, _ = metrics.precision_recall_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(pr_r, pr_p, label=f'PR-AUC = {metrics.average_precision_score(y_test, p_test):.3f}')
    ax.axhline(y_test.mean(), color='k', ls='--', lw=0.8, label='случайная модель')
    if pred.sum() > 0:
        ax.scatter(rec, prec, color='red', s=60, zorder=5, label='текущий порог')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR-кривая')
    ax.legend(loc='upper right')
    fig_pr.pyplot(fig)

# ---------- 4.3 Пакетное предсказание ----------
with tab3:
    st.subheader('Загрузите CSV с характеристиками вин')
    st.write(f'Порог берётся из слайдера в сайдбаре: **{st.session_state["threshold"]:.2f}**. '
             f'Нужные столбцы: `{", ".join(FEATURES)}`.')
    file = st.file_uploader('CSV-файл', type='csv')
    if file is not None:
        try:
            data = pd.read_csv(file)
        except Exception as e:
            st.error(f'Не удалось прочитать файл как CSV: {e}')
            data = None
        if data is not None:
            missing = [f for f in FEATURES if f not in data.columns]
            if missing:
                st.error(f'В файле не хватает столбцов: {", ".join(missing)}')
            else:
                try:
                    proba = model.predict_proba(data[FEATURES].astype(float))[:, 1]
                    out = data.copy()
                    out['proba_good'] = proba.round(4)
                    out['in_selection'] = (proba >= st.session_state['threshold']).astype(int)
                    st.write(f'В подборку попало {out["in_selection"].sum()} из {len(out)} вин.')
                    st.dataframe(out, width="stretch")
                    st.download_button('Скачать результат (CSV)', out.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv')
                except Exception as e:
                    st.error(f'Ошибка при предсказании (проверьте, что значения числовые): {e}')

# ---------- 4.4 Что выучила модель ----------
with tab4:
    st.subheader('Коэффициенты логистической регрессии (признаки масштабированы)')
    coefs = pd.Series(model[-1].coef_.ravel(), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coefs.index, coefs.values, color=['tab:red' if v < 0 else 'tab:green' for v in coefs])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Коэффициент')
    st.pyplot(fig)
    st.write(f'**За хорошее вино** сильнее всего «голосуют»: {", ".join(coefs.index[::-1][:3])}.')
    st.write(f'**Против** сильнее всего «голосуют»: {", ".join(coefs.index[:3])}.')
    st.caption('Так как признаки стандартизованы, коэффициенты можно сравнивать между собой: '
               'чем больше модуль, тем сильнее признак влияет на вероятность.')
