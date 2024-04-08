import torch
from transformers import AutoProcessor,MusicgenForConditionalGeneration
from os import environ
from io import BytesIO
from scipy.io.wavfile import write
from pygame import mixer
from queue import Queue
from threading import Thread


MODEL_ID="facebook/musicgen-small"
TEXT=[
    #"An airy piano melody accompanied by gentle strings, creating a serene ambiance perfect for unwinding, studying, or finding focus."
    #"Upbeat acoustic guitar tune with catchy whistling melodies and a relaxed vibe."
    #"strange and out of tune"
    "lo-fi music with a soothing melody"
]
AUDIO_LEN=30
CUT_OFF_SEC=3
environ["PULSE_SERVER"]="192.168.156.212"
#environ["CUDA_LAUNCH_BLOCKING"]="1"
MAX_QUEUE_SIZE=10

#import vars
token_len=int(AUDIO_LEN*51.2)
device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
process=AutoProcessor.from_pretrained(
    MODEL_ID
)
model=MusicgenForConditionalGeneration.from_pretrained(
    MODEL_ID
).to(device)
sr=model.config.audio_encoder.sampling_rate

audio_queue=Queue(maxsize=MAX_QUEUE_SIZE)
def worker(audio,queue):
    while True:
        audio=model.generate(
            **process(
                audio=audio[-sr*CUT_OFF_SEC:],
                text=TEXT,
                sampling_rate=sr,
                padding=True,
                return_tensors="pt"
            ).to(device),
            max_new_tokens=token_len
        )[0].cpu()[0][sr*CUT_OFF_SEC:sr*30]
        queue.put(audio)

'''start genrating the first audio'''

audio0=model.generate(
    **process(
        text=TEXT,
        padding=True,
        return_tensors="pt"
    ).to(device),
    max_new_tokens=token_len
)[0].cpu()[0][:sr*AUDIO_LEN]

torch.cuda.empty_cache()
audio_queue.put(audio0)

thread=Thread(
    target=worker,
    args=(
        audio0,
        audio_queue
    )
)
thread.daemon=True
thread.start()

mixer.init()
chan=mixer.Channel(0)

while True:
    wav=BytesIO()
    write(
        filename=wav,
        rate=sr,
        data=audio_queue.get().numpy()
    )
    chan.queue(mixer.Sound(wav))



