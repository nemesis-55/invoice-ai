# import json
# import pandas as pd
# import base64
# import os

# INPUT_JSON = "data/train_data copy.json"
# OUTPUT_PARQUET = "data/train_data_parquet.parquet"

# def encode_image(image_path):
#     try:
#         with open(image_path, "rb") as img_file:
#             return base64.b64encode(img_file.read()).decode("utf-8")
#     except Exception:
#         return None

# def main():
#     with open(INPUT_JSON, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     rows = []
#     for item in data:
#         image_path = item.get("image", "")
#         image_abs_path = os.path.join(os.path.dirname(INPUT_JSON), image_path)
#         image_b64 = encode_image(image_abs_path)
#         conversations = item.get("conversations", [])
#         user_msg = next((conv["value"] for conv in conversations if conv["from"] == "user"), "")
#         assistant_msg = next((conv["value"] for conv in conversations if conv["from"] == "assistant"), "")
#         rows.append({
#             "id": item.get("id", ""),
#             "image_path": image_path,
#             "image_b64": image_b64,
#             "user_msg": user_msg,
#             "assistant_msg": assistant_msg
#         })

#     df = pd.DataFrame(rows)
#     df.to_parquet(OUTPUT_PARQUET, index=False)
#     print(f"✅ Saved to {OUTPUT_PARQUET}")

# if __name__ == "__main__":
#     main()

import json
import os

# Input and output file paths
INPUT_JSON = "data/axolotl_format.json"
OUTPUT_JSON = "data/qwen_format.json"

def convert_to_qwen_format(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []

    for idx, sample in enumerate(data):
        messages = sample.get("messages", [])
        qwen_item = {
            "id": f"identity_{idx}",
            "conversations": []
        }

        # Extract user/assistant pairs
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", [])

            # Convert content to text string
            text_parts = []
            for part in content:
                if part["type"] == "image":
                    text_parts.append(f"<img>{part['path']}</img>")
                elif part["type"] == "text":
                    text_parts.append(part["text"].strip())

            text_value = "\n".join(text_parts).strip()

            # Skip system messages
            if role == "system":
                continue

            # Map roles
            from_role = "user" if role == "user" else "assistant"

            qwen_item["conversations"].append({
                "from": from_role,
                "value": text_value
            })

        output.append(qwen_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(output)} samples to Qwen-VL format.")
    print(f"💾 Output saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    convert_to_qwen_format(INPUT_JSON, OUTPUT_JSON)
