import wave

with open('test.wav', 'w') as nf:
    fp = wave.open(nf)
    fp.setnchannels(2)
    fp.setframerate(44100)
    fp.setsampwidth(2)
    fp.close()