"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    import pandas as pd

    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")

    women_df = df[df["Gender"] == "Female"]
    men_df = df[df["Gender"] == "Male"]

    women_count = len(women_df)
    men_count = len(men_df)

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    # Считаем корреляцию Пирсона каждого признака с MRI_Count
    women_corr = {
        feat: float(women_df[feat].corr(women_df["MRI_Count"], method="pearson"))
        for feat in features
    }
    men_corr = {
        feat: float(men_df[feat].corr(men_df["MRI_Count"], method="pearson"))
        for feat in features
    }

    # Находим признак с наибольшим модулем корреляции среди обеих групп
    all_corrs = []
    for feat in features:
        all_corrs.append((feat, abs(women_corr[feat])))
        all_corrs.append((feat, abs(men_corr[feat])))

    strongest_mri_feature = max(all_corrs, key=lambda item: item[1])[0]

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_mri_feature,
    )


if __name__ == "__main__":
    result = analyze_brain_correlations(BrainDataInput("brainsize.txt"))
    print("=" * 40)
    print("РЕЗУЛЬТАТЫ КОРРЕЛЯЦИОННОГО АНАЛИЗА:")
    print("=" * 40)
    print("Мужчин в выборке:", result.men_count)
    print("Женщин в выборке:", result.women_count)
    print("\nКорреляции признаков с MRI_Count у женщин:")
    for feat, val in result.women_mri_correlation.items():
        print(f"  {feat}: {val:.4f}")
    print("\nКорреляции признаков с MRI_Count у мужчин:")
    for feat, val in result.men_mri_correlation.items():
        print(f"  {feat}: {val:.4f}")
    print("\nПризнак с самой сильной связью с объемом мозга (по модулю):", result.strongest_mri_feature)
    print("=" * 40)
