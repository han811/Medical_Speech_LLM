from model import language, speech


language_model = language.make("llama")()
speech_encoder = speech.make("hubert")()

print(language_model)
print(speech_encoder)
