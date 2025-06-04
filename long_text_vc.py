import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS


AUDIO_PROMPT_PATH = "voices/ballen.wav"


def chunk_text(text, initial_chunk_size=400):
    """Split text into chunks at sentence boundaries with dynamic sizing."""
    sentences = text.replace("\n", " ").split(".")
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_size = initial_chunk_size

    for sentence in sentences:
        if not sentence.strip():
            continue  # Skip empty sentences

        sentence = sentence.strip() + "."
        sentence_size = len(sentence)

        # If a single sentence is too long, split it into smaller pieces
        if sentence_size > chunk_size:
            words = sentence.split()
            current_piece = []
            current_piece_size = 0

            for word in words:
                word_size = len(word) + 1  # +1 for space
                if current_piece_size + word_size > chunk_size:
                    if current_piece:
                        chunks.append(" ".join(current_piece).strip() + ".")
                    current_piece = [word]
                    current_piece_size = word_size
                else:
                    current_piece.append(word)
                    current_piece_size += word_size

            if current_piece:
                chunks.append(" ".join(current_piece).strip() + ".")
            continue

        # Start new chunk if current one would be too large
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_chunked_audio(text, output_file="output.wav", chunk_size=400):
    """Generate TTS audio for text chunks and merge into single file."""

    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Initialize the TTS model
    model = ChatterboxTTS.from_pretrained(device=device)

    # Split text into chunks
    chunks = chunk_text(text, chunk_size)
    print(f"Text split into {len(chunks)} chunks")

    # Generate audio for each chunk
    audio_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        wav = model.generate(
            chunk,
            audio_prompt_path=AUDIO_PROMPT_PATH,
            exaggeration=0.85,
            cfg_weight=0.55,
        )
        audio_chunks.append(wav)

    # Merge all audio chunks
    if audio_chunks:
        merged_audio = torch.cat(audio_chunks, dim=-1)

        # Save the merged audio
        ta.save(output_file, merged_audio, model.sr)
        print(f"Audio saved to {output_file}")

        return merged_audio
    else:
        print("No audio chunks generated")
        return None


# Example usage
if __name__ == "__main__":
    # Example text (you can replace this with your longer text)

    text = """
    This combined script does the following:

Preserves your chunking logic - The chunk_text function remains exactly as you had it, splitting text at sentence boundaries with dynamic sizing.
Device detection - Automatically detects and uses the best available device (CUDA, MPS, or CPU).
Chunked audio generation - The generate_chunked_audio function:

Splits the input text into chunks
Generates TTS audio for each chunk individually
Concatenates all audio chunks using torch.cat()
Saves the merged result to a single audio file


Progress tracking - Shows which chunk is being processed and total progress.

The main advantages of this approach:

Memory efficient - Processes text in manageable chunks rather than all at once
Handles long texts - Can process very long documents without overwhelming the TTS model
Preserves sentence boundaries - Your chunking logic ensures natural breaks
Single output file - All chunks are merged into one continuous audio file

You can adjust the chunk_size parameter based on your TTS model's optimal input length and available memory. Smaller chunks use less memory but may have more processing overhead, while larger chunks are more efficient but require more memory.
    """

    # Generate audio with custom chunk size and output file
    generate_chunked_audio(text, "merged_output.wav", chunk_size=300)
