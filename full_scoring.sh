#!/bin/bash

# Список моделей
models=(
    "jinaai/jina-embeddings-v3"
    "Alibaba-NLP/gte-large-en-v1.5"
    "jxm/cde-small-v1"
    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"
)

# Путь к скрипту
script_path="evaluate/scripts/ft.sh"

# Аргументы для скрипта
data_path="data/_downstream_data"
metrics="metrics"

# Перебор моделей и выполнение скрипта
for model in "${models[@]}"
do
    echo "Running script for model: $model"
    bash "$script_path" "$model" "$data_path" "$metrics"
    
    # Проверка статуса выполнения последней команды
    if [ $? -ne 0 ]; then
        echo "Error occurred while running for model: $model"
        exit 1
    fi
done

echo "All models processed successfully."
