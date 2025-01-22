#!/bin/bash

# Список моделей
models=(
    "jinaai jina-embeddings-v3"
    "dunzhang stella_en_400M_v5"
    "HIT-TMG KaLM-embedding-multilingual-mini-instruct-v1"
    "jxm cde-small-v1"
)

# Путь к скрипту
script_path="evaluate/scripts/sim.sh"

# Аргументы для скрипта
data_path="data/_downstream_data"
metrics="metrics"

# Перебор моделей и выполнение скрипта
for model in "${models[@]}"
do
    # Разделение строки модели на две переменные: провайдер и имя модели
    provider=$(echo $model | cut -d' ' -f1)
    model_name=$(echo $model | cut -d' ' -f2)

    echo "Running script for provider: $provider, model: $model_name"
    
    bash "$script_path" "$provider" "$model_name" "$data_path" "$metrics"

    # Проверка статуса выполнения последней команды
    if [ $? -ne 0 ]; then
        echo "Error occurred while running for provider: $provider, model: $model_name"
        echo "Skipping to the next model."
        continue
    fi

echo "Successfully processed provider: $provider, model: $model_name"
done

echo "All models processed."
