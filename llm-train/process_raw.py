import os
import json

def process(directory, chunk_size=512):
    raw_directory = f"{directory}/raw/"
    output_file = f"{directory}/corpus.json"
    chunks = []
    files = os.listdir(raw_directory)
    for file_name in files:
        if file_name.endswith(".txt"):
            file_path = os.path.join(raw_directory, file_name)
            with open(file_path, 'r') as file:
                chunk_id = 0
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break

                    chunk_data = {
                        'name': file_name,
                        'chunk_id': chunk_id,
                        'text': chunk
                    }
                    chunks.append(chunk_data)
                    chunk_id += 1

    with open(output_file, 'w') as json_file:
        json.dump(chunks, json_file, ensure_ascii=False, indent=4)

    print("Process Done!")

corpus_directory = "./datasets/corpus/"

process(corpus_directory)
