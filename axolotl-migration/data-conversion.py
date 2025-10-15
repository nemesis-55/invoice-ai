import json
import os

# Input and output file paths
INPUT_JSON = "data/train_data.json"
OUTPUT_JSON = "data/axolotl_format.json"

# Default system instruction
SYSTEM_PROMPT = "You are a helpful assistant that extracts structured data from invoice images."

def convert_to_axolotl_format(input_path: str, output_path: str):
    # Load the input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []

    for item in data:
        image_path = item.get("image", "")
        conversations = item.get("conversations", [])

        # Extract user and assistant parts
        user_msg = next((conv["content"] for conv in conversations if conv["role"] == "user"), None)
        assistant_msg = next((conv["content"] for conv in conversations if conv["role"] == "assistant"), None)

        if not user_msg or not assistant_msg:
            continue  # skip incomplete data

        # Clean user text (remove "<image>" tag and extra newlines)
        user_text = user_msg.replace("<image>", "").strip()

        # Build the message structure
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_msg.strip()}
                ]
            }
        ]

        output.append({"messages": messages})

    # Save the transformed data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(output)} records to Axolotl format.")
    print(f"💾 Output saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    convert_to_axolotl_format(INPUT_JSON, OUTPUT_JSON)
