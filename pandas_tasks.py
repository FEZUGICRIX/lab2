"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)

    row_count = len(df)
    missing_by_column = df.isna().sum().to_dict()
    adults_over_30_count = int((df["Age"] > 30).sum())

    mean_age_by_pclass = {
        int(pclass): float(mean_age)
        for pclass, mean_age in df.groupby("Pclass")["Age"].mean().items()
    }

    survival_rate_by_pclass = {
        int(pclass): float(rate)
        for pclass, rate in df.groupby("Pclass")["Survived"].mean().items()
    }

    highest_fares = [float(f) for f in df["Fare"].nlargest(5).tolist()]

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )


if __name__ == "__main__":
    result = analyze_titanic(TitanicInput("titanic.csv"))
    print("=" * 40)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА ТИТАНИКА:")
    print("=" * 40)
    print("1. Всего пассажиров:", result.row_count)
    print("2. Пропуски в столбцах:", result.missing_by_column)
    print("3. Пассажиров старше 30 лет:", result.adults_over_30_count)
    print("4. Средний возраст по классам кают:", result.mean_age_by_pclass)
    print("5. Доля выживших по классам кают:", result.survival_rate_by_pclass)
    print("6. Топ-5 самых дорогих билетов:", result.highest_fares)
    print("=" * 40)
