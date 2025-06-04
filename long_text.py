import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS


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
        wav = model.generate(chunk)
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
    Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill. 
    The battle was intense, with each champion using their unique abilities to outmaneuver the opposition. 
    Ezreal's precise skill shots found their mark, while Jinx's explosive rockets cleared waves of minions. 
    Ahri charmed key targets, Yasuo danced through enemy lines with his wind wall, and Teemo's mushrooms 
    created a deadly maze that the enemies couldn't escape.
    """

    text = """
        The Gentoo Handbook
        Architectures

        Gentoo Linux is available for many computer architectures.

        An instruction set architecture (ISA) (Wikipedia) or architecture for short is a family of CPUs (processors) who support the same instructions. The two most prominent architectures in the desktop world are the x86 architecture and the x86_64 architecture (for which Gentoo uses the amd64 notation). But many other architectures exist, such as sparc, ppc (the PowerPC family), mips, arm, etc...

        A distribution as versatile as Gentoo supports many architectures. Below is a quick summary of the supported architectures and the abbreviation used in Gentoo. Most people that do not know the architecture of their PC system are likely interested in amd64.
        Viewing the Handbook

        The list below gives a high-level overview of the architectures supported by various Gentoo Linux projects. It is important to choose the correct architecture before proceeding with the associated section of a Handbook. Be sure to verify the CPU's architecture before moving onward.

        The main link for each Handbook provides a section-by-section view for each of the four chapters. The Handbook project recommends this section-by-section view when installing Gentoo.

        Alternatively, a single page per-chapter view is provided for readers who wish to view a single chapter at a time. This view is useful for easily searching a chapter or for printing.
    """

    # Generate audio with custom chunk size and output file
    generate_chunked_audio(text, "merged_output.wav", chunk_size=300)
