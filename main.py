from pipelines.VoicePipeline import VoicePipeline
from utils.audio import Player, load_wav

SAMPLE = "samples/long.wav"
PLAY_AUDIO = True
REPEATS = 5


def main():
    pipe = VoicePipeline()
    pipe.warmup()
    player = Player(pipe.tts.sample_rate) if PLAY_AUDIO else None

    audio = load_wav(SAMPLE)
    for i in range(REPEATS):
        pipe.llm.reset()
        turn = pipe(audio, player=player)
        print(
            f"run {i}: first_audio {turn.timings['first_audio_ms']:.0f} ms | "
            f"stt {turn.timings['stt_ms']:.0f} | ttft {turn.timings['llm_ttft_ms']:.0f}"
        )
        if player:
            player.wait()


if __name__ == "__main__":
    main()
