from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree


TIME_VALUE = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
SUBTITLE_TIMING_RE = re.compile(
    rf"^\s*(?P<start>{TIME_VALUE})\s*-->\s*(?P<end>{TIME_VALUE})"
)
TURN_RE = re.compile(
    rf"""
    ^\s*
    (?:[-*]\s*)?
    (?:\[?(?P<timestamp>{TIME_VALUE})\]?\s*(?:[-–—]\s*)?)?
    (?P<speaker>[A-Za-zА-Яа-яЁё0-9 _.'"-]{{1,80}}?)
    \s*[:：]\s*
    (?P<text>.*)
    \s*$
    """,
    re.VERBOSE,
)
TIMED_SPEAKER_ONLY_RE = re.compile(
    rf"""
    ^\s*
    (?:[-*]\s*)?
    \[?(?P<timestamp>{TIME_VALUE})\]?
    \s*(?:[-–—]\s*)?
    (?P<speaker>[A-Za-zА-Яа-яЁё0-9 _.'"-]{{1,80}})
    \s*$
    """,
    re.VERBOSE,
)
SPEAKER_ONLY_RE = re.compile(
    r"""^\s*(?P<speaker>[A-Za-zА-Яа-яЁё0-9 _.'"-]{1,80})\s*$""",
    re.VERBOSE,
)
FENCED_BLOCK_RE = re.compile(r"```(?:text|txt|transcript)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

SELLER_HINTS = {
    "продавец",
    "seller",
    "ии",
    "ai",
    "бот",
    "ассистент",
    "оппонент",
}
BUYER_HINTS = {
    "покупатель",
    "заказчик",
    "клиент",
    "участник",
    "buyer",
    "customer",
    "client",
    "participant",
}
SKIP_LINES = {"webvtt", "стенограмма", "стенограмма диалога", "transcript"}


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_docx_text(path)
    if suffix in {".txt", ".md", ".srt", ".vtt"}:
        return read_text_file(path)
    raise ValueError(
        f"Unsupported file type '{path.suffix}'. Use .docx, .txt, .md, .srt, or .vtt."
    )


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ElementTree.fromstring(document_xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []

    for paragraph in root.iter(f"{namespace}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            tag = node.tag.split("}", 1)[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
            elif tag in {"br", "cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def select_transcript_region(text: str) -> str:
    fenced_blocks = FENCED_BLOCK_RE.findall(text)
    for block in fenced_blocks:
        if count_turn_like_lines(block.splitlines()) >= 2:
            return block

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "стенограмм" in line.lower() or line.strip().lower() == "transcript":
            tail = "\n".join(lines[index + 1 :])
            if count_turn_like_lines(tail.splitlines()) >= 2:
                return tail

    return text


def count_turn_like_lines(lines: Iterable[str]) -> int:
    return sum(1 for line in lines if TURN_RE.match(line.strip()))


def parse_transcript(raw_text: str, *, merge_consecutive: bool = True) -> list[dict[str, object]]:
    text = select_transcript_region(raw_text)
    turns: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    pending_timestamp: str | None = None
    pending_speaker: str | None = None

    for raw_line in text.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue
        if should_skip_line(line):
            continue

        subtitle_match = SUBTITLE_TIMING_RE.match(line)
        if subtitle_match:
            pending_timestamp = normalize_timestamp(subtitle_match.group("start"))
            continue

        turn_match = TURN_RE.match(line)
        if turn_match:
            timestamp = turn_match.group("timestamp") or pending_timestamp
            speaker = normalize_speaker_label(turn_match.group("speaker"))
            turn_text = normalize_text(turn_match.group("text"))
            current = {
                "timestamp": normalize_timestamp(timestamp) if timestamp else None,
                "speaker_label": speaker,
                "role": "unknown",
                "text": turn_text,
            }
            turns.append(current)
            pending_timestamp = None
            pending_speaker = None
            continue

        timed_speaker_match = TIMED_SPEAKER_ONLY_RE.match(line)
        if timed_speaker_match and looks_like_speaker(timed_speaker_match.group("speaker")):
            pending_timestamp = normalize_timestamp(timed_speaker_match.group("timestamp"))
            pending_speaker = normalize_speaker_label(timed_speaker_match.group("speaker"))
            continue

        speaker_only_match = SPEAKER_ONLY_RE.match(line)
        if speaker_only_match and looks_like_speaker(speaker_only_match.group("speaker")):
            pending_speaker = normalize_speaker_label(speaker_only_match.group("speaker"))
            continue

        continuation = normalize_text(line)
        if pending_speaker:
            current = {
                "timestamp": pending_timestamp,
                "speaker_label": pending_speaker,
                "role": "unknown",
                "text": continuation,
            }
            turns.append(current)
            pending_timestamp = None
            pending_speaker = None
        elif current:
            current["text"] = join_text(str(current["text"]), continuation)

    turns = [turn for turn in turns if str(turn.get("text", "")).strip()]
    apply_role_inference(turns)

    if merge_consecutive:
        turns = merge_adjacent_same_speaker(turns)

    for index, turn in enumerate(turns, start=1):
        turn["turn_index"] = index

    return turns


def should_skip_line(line: str) -> bool:
    normalized = line.strip().lower().strip("#").strip()
    if normalized in SKIP_LINES:
        return True
    if normalized.isdigit():
        return True
    return False


def clean_line(line: str) -> str:
    return (
        line.replace("\ufeff", "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .strip()
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return normalize_text(f"{left} {right}")


def normalize_speaker_label(label: str) -> str:
    return normalize_text(label).strip(" -–—")


def normalize_timestamp(timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    timestamp = timestamp.replace(",", ".")
    main = timestamp.split(".", 1)[0]
    parts = main.split(":")
    if len(parts) == 2:
        minute, second = parts
        return f"{int(minute):02d}:{int(second):02d}"
    if len(parts) == 3:
        hour, minute, second = parts
        if int(hour) == 0:
            return f"{int(minute):02d}:{int(second):02d}"
        return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
    return timestamp


def looks_like_speaker(label: str) -> bool:
    normalized = normalize_speaker_key(label)
    if normalized in SELLER_HINTS or normalized in BUYER_HINTS:
        return True
    if len(normalized.split()) <= 3 and not re.search(r"[.!?,;]", normalized):
        return True
    return False


def normalize_speaker_key(label: str) -> str:
    return normalize_speaker_label(label).lower().replace("ё", "е")


def infer_role(label: str) -> str:
    key = normalize_speaker_key(label)
    if key in SELLER_HINTS or any(hint in key.split() for hint in SELLER_HINTS):
        return "seller"
    if key in BUYER_HINTS or any(hint in key.split() for hint in BUYER_HINTS):
        return "buyer"
    return "unknown"


def apply_role_inference(turns: list[dict[str, object]]) -> None:
    speakers = list(dict.fromkeys(str(turn["speaker_label"]) for turn in turns))
    role_by_speaker = {speaker: infer_role(speaker) for speaker in speakers}

    if len(speakers) == 2:
        first, second = speakers
        if role_by_speaker[first] == "seller" and role_by_speaker[second] == "unknown":
            role_by_speaker[second] = "buyer"
        elif role_by_speaker[first] == "buyer" and role_by_speaker[second] == "unknown":
            role_by_speaker[second] = "seller"
        elif role_by_speaker[second] == "seller" and role_by_speaker[first] == "unknown":
            role_by_speaker[first] = "buyer"
        elif role_by_speaker[second] == "buyer" and role_by_speaker[first] == "unknown":
            role_by_speaker[first] = "seller"

    for turn in turns:
        turn["role"] = role_by_speaker.get(str(turn["speaker_label"]), "unknown")


def merge_adjacent_same_speaker(turns: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for turn in turns:
        if (
            merged
            and merged[-1]["speaker_label"] == turn["speaker_label"]
            and merged[-1]["role"] == turn["role"]
        ):
            merged[-1]["text"] = join_text(str(merged[-1]["text"]), str(turn["text"]))
            continue
        merged.append(dict(turn))
    return merged


def build_output(path: Path, turns: list[dict[str, object]]) -> dict[str, object]:
    speaker_counts = Counter(str(turn["speaker_label"]) for turn in turns)
    speaker_roles: dict[str, str] = {}
    for turn in turns:
        speaker_roles.setdefault(str(turn["speaker_label"]), str(turn["role"]))

    return {
        "schema_version": "transcript_json_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transcript": {
            "source_format": "speaker_turns",
            "source_file": str(path),
            "language": "ru",
            "turn_count": len(turns),
            "speakers": [
                {
                    "speaker_label": speaker,
                    "role": speaker_roles.get(speaker, "unknown"),
                    "utterance_count": count,
                }
                for speaker, count in speaker_counts.items()
            ],
            "utterances": turns,
        },
    }


def choose_input_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local Python build
        raise RuntimeError("tkinter is required for the file picker window.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    filename = filedialog.askopenfilename(
        title="Choose transcript file",
        filetypes=[
            ("Transcript files", "*.docx *.txt *.md *.srt *.vtt"),
            ("Word documents", "*.docx"),
            ("Text files", "*.txt *.md *.srt *.vtt"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(filename) if filename else None


def confirm_overwrite(path: Path) -> bool:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return False

    root = tk.Tk()
    root.withdraw()
    root.update()
    answer = messagebox.askyesno(
        "Overwrite JSON?",
        f"The file already exists:\n{path}\n\nOverwrite it?",
    )
    root.destroy()
    return bool(answer)


def show_done_message(path: Path, turn_count: int) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return

    root = tk.Tk()
    root.withdraw()
    root.update()
    messagebox.showinfo(
        "Transcript converted",
        f"Saved JSON:\n{path}\n\nUtterances: {turn_count}",
    )
    root.destroy()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a seller/buyer dialogue transcript into JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Transcript file. If omitted, a file picker window is opened.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file. Defaults to '<input_stem>_transcript.json'.",
    )
    parser.add_argument(
        "--no-merge-consecutive",
        action="store_true",
        help="Keep consecutive fragments from the same speaker as separate turns.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Overwrite an existing output file without asking.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    launched_from_gui = args.input is None

    input_path = args.input or choose_input_file()
    if input_path is None:
        return 0
    input_path = input_path.resolve()

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or input_path.with_name(f"{input_path.stem}_transcript.json")
    output_path = output_path.resolve()

    if output_path.exists() and not args.yes:
        if launched_from_gui:
            if not confirm_overwrite(output_path):
                return 0
        else:
            print(
                f"Output file already exists: {output_path}\n"
                "Use --yes to overwrite or --output to choose another path.",
                file=sys.stderr,
            )
            return 1

    raw_text = extract_text(input_path)
    turns = parse_transcript(raw_text, merge_consecutive=not args.no_merge_consecutive)
    if not turns:
        print("No speaker turns were found in the selected file.", file=sys.stderr)
        return 1

    payload = build_output(input_path, turns)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {output_path}")
    print(f"Utterances: {len(turns)}")

    if launched_from_gui:
        show_done_message(output_path, len(turns))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
