from pathlib import Path
import re

RAW_PATH = Path("data/raw/shakespeare.txt")
OUT_PATH = Path("data/processed/chat_data.txt")

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


def is_speaker_line(line):
    """
    Detect lines like:
    PETRUCHIO:
    KING HENRY:
    """
    return bool(re.match(r"^[A-Z][A-Z\s]+:$", line))


def clean_line(line):
    line = line.strip()
    line = re.sub(r"\[.*?\]", "", line)  # remove [stage directions]
    return line.strip()


def build_chat_data():
    text = RAW_PATH.read_text(encoding="utf-8").splitlines()

    dialogues = []
    current_speaker = None
    current_utterance = []

    for line in text:
        line = clean_line(line)

        if not line:
            continue

        if is_speaker_line(line):
            if current_speaker and current_utterance:
                dialogues.append(
                    " ".join(current_utterance).strip()
                )
            current_speaker = line
            current_utterance = []
        else:
            if current_speaker:
                current_utterance.append(line)

    # Add last utterance
    if current_speaker and current_utterance:
        dialogues.append(" ".join(current_utterance).strip())

    print(f"Extracted {len(dialogues)} dialogue lines")

    # Build chat pairs
    chat_samples = []

    for i in range(len(dialogues) - 1):
        user_text = dialogues[i]
        assistant_text = dialogues[i + 1]

        sample = (
            f"{USER_TOKEN} {user_text}\n"
            f"{ASSISTANT_TOKEN} {assistant_text}\n"
            f"{END_TOKEN}\n\n"
        )
        chat_samples.append(sample)

    OUT_PATH.write_text("".join(chat_samples), encoding="utf-8")
    print(f"Saved chat data to {OUT_PATH}")


if __name__ == "__main__":
    build_chat_data()
