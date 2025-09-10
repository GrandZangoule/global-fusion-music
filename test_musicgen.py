from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def main():
    print("Loading MusicGen model... (first run will download weights)")
    # Start with the smaller model for CPU performance
    model = MusicGen.get_pretrained("facebook/musicgen-small")

    # Configure generation settings
    model.set_generation_params(
        duration=10,      # length in seconds
        top_k=250,
        temperature=1.0
    )

    # Prompt for generation
    prompt = "West African highlife with guitars and shakers, cheerful, 120 BPM"
    print(f"Generating music for prompt: {prompt}")
    wav = model.generate([prompt], progress=True)

    # Save to file
    audio_write("musicgen_test", wav[0].cpu(), model.sample_rate, strategy="loudness")
    print("✅ Done! Saved as musicgen_test.wav")

if __name__ == "__main__":
    main()