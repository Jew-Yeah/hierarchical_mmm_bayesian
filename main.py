import argparse
import glob
import os
from pipelines.forecast_channel import forecast_channel  # Импортируем forecast_channel из папки pipelines

def get_user_input():
    """
    Функция для получения пользовательского ввода в консольном интерфейсе.
    """
    try:
        print("Выберите датасет для работы:\n")
        # Получить все папки в текущей директории
        folders = glob.glob("./data/*/")
        folder_names = [os.path.basename(f.rstrip('\\').rstrip('/')) for f in folders]
        for elem in folder_names:
            print(f"- {elem}")
        dataset_name = input(f"\nВведите имя датасета (по умолчанию '{folder_names[0]}'): ") or folder_names[0]
        func_options = {
            1:"forecast_channel",
            2:"optimize_budget",
            3:"backtest_compare"
        }

        print("\nВыберите пайплайн для выполнения:\n")
        for key in func_options.keys():
            print(f"{key} - {func_options[key]}")

        pipeline_choice = int(input("\nВведите номер пайплайна (по умолчанию '1'): ") or 1)

        if pipeline_choice in func_options.keys():
            return dataset_name, func_options[pipeline_choice]
    except:
        print("Неверный выбор пайплайна. Завершаем работу.")
        return None, None

def main():
    pipelines_chosen = {
        "forecast_channel":forecast_channel,
    }
    dataset_name, func_chosen = get_user_input()
    result = pipelines_chosen[func_chosen](dataset_name)

    if not(result is None):
        print(result)

if __name__ == "__main__":
    main()
